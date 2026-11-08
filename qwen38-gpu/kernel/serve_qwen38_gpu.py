#!/usr/bin/env python3
"""
Qwen3.8-27B on a free Kaggle GPU (2x Tesla T4, 15.36 GB each) via llama.cpp.

Why llama.cpp and not vLLM: the T4 is Turing (sm_75) and the model is bf16;
llama.cpp runs GGUF quantized weights in fp16 on Turing without the BF16
tensor-core path vLLM assumes. llama.cpp also has native LLM_ARCH_QWEN35.

Flow: prebuilt CUDA llama.cpp -> GGUF from HF -> llama-server -> cloudflared
-> READY banner on ntfy. Same contract as the TPU recipe, so the gateway can
drive it unchanged.

CFG is replaced by launch.py (or passed via serve_config.json):
  {"ntfy_topic": "...", "api_key": "...", "keepalive_min": 480, ...}
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CFG = None  # __LAUNCHER_CONFIG__

DEFAULTS = {
    "model_repo": "unsloth/Qwen3.8-27B-GGUF",
    "model_file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
    "model_bytes": 16464440224,
    "llama_release": "b12202",
    "llama_asset": "llama-b12202-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart_asset": "cudart-llama-b12202-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "port": 8000,
    "ctx": 32768,
    "parallel": 2,
    "api_key": "",
    "ntfy_topic": "",
    "keepalive_min": 480,
    "served_model_name": "qwen3.8-27b",
    "self_test": True,
    "tensor_split": "",
}
CFG = {**DEFAULTS, **(CFG or {})}
_cfg_file = Path("serve_config.json")
if _cfg_file.exists():
    CFG.update(json.loads(_cfg_file.read_text()))
if not CFG["api_key"]:
    import secrets
    CFG["api_key"] = "sk-" + secrets.token_hex(16)

WORK = Path("/tmp/ktl-gpu")
WORK.mkdir(parents=True, exist_ok=True)
PORT = CFG["port"]


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + str(msg), flush=True)


def publish(phase, **extra):
    body = {"phase": phase, "ts": int(time.time()), **extra}
    log(f"[event] {phase} {json.dumps(extra)[:300]}")
    topic = CFG.get("ntfy_topic")
    if not topic:
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Title": f"ktl-gpu {phase}"},
        )
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        log(f"(ntfy failed: {e})")


def download(url, dest, expected=None, label="", attempts=3):
    dest = Path(dest)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        log(f"   {label or dest.name} already present ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    urls = [url] if isinstance(url, str) else list(url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for u in urls:
        for attempt in range(1, attempts + 1):
            try:
                log(f"   downloading {label or dest.name} <- {u[:90]} (try {attempt}/{attempts})")
                t0 = time.time()
                done = 0
                last = 0
                req = urllib.request.Request(u, headers={"User-Agent": "ktl-gpu"})
                with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    while True:
                        chunk = r.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if done - last > 2e9:
                            last = done
                            pct = f"{100 * done / total:.0f}%" if total else ""
                            log(f"     {done / 1e9:.1f} GB {pct} ({done / 1e6 / max(1e-9, time.time() - t0):.0f} MB/s)")
                if expected and done != expected:
                    raise RuntimeError(f"short read: {done} != {expected}")
                tmp.replace(dest)
                log(f"   done {dest.name}: {dest.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s")
                return dest
            except Exception as e:
                last_err = e
                log(f"   download failed ({type(e).__name__}: {e})")
                try:
                    tmp.unlink()
                except OSError:
                    pass
                if attempt < attempts:
                    time.sleep(5 * attempt)
    raise RuntimeError(f"download failed after {len(urls) * attempts} attempts: {last_err}")


GH = "https://github.com/ggml-org/llama.cpp/releases/download"
GH_MIRROR = "https://gh-proxy.com/https://github.com/ggml-org/llama.cpp/releases/download"


def release_urls(release, asset):
    return [
        f"{GH}/{release}/{asset}",
        f"{GH_MIRROR}/{release}/{asset}",
    ]


def remote_size(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ktl-gpu", "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range") or ""
        if "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None


def download_ranged(url, dest, expected=None, label="", chunk=16 * 1024 * 1024, attempts=6):
    """Chunked GET. GitHub's release edge often 504s a long single stream but
    answers Range requests; 16 MB chunks with retries are reliable."""
    dest = Path(dest)
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        log(f"   {label or dest.name} already present ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    urls = [url] if isinstance(url, str) else list(url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for u in urls:
        try:
            total = remote_size(u)
        except Exception as e:
            last_err = e
            log(f"   size probe failed on {u[:80]}: {e}")
            continue
        if not total:
            continue
        if expected and total != expected:
            log(f"   {u[:60]}: size {total} != expected {expected}, trying next mirror")
            continue
        log(f"   ranged download {label or dest.name}: {total / 1e9:.2f} GB in {chunk // 2**20} MB chunks")
        t0 = time.time()
        offset = 0
        ok = True
        with open(tmp, "wb") as f:
            while offset < total:
                end = min(offset + chunk, total) - 1
                for attempt in range(1, attempts + 1):
                    try:
                        req = urllib.request.Request(u, headers={
                            "User-Agent": "ktl-gpu", "Range": f"bytes={offset}-{end}"})
                        with urllib.request.urlopen(req, timeout=90) as r:
                            data = r.read()
                        if len(data) != end - offset + 1:
                            raise RuntimeError(f"short chunk {len(data)} != {end - offset + 1}")
                        f.write(data)
                        offset += len(data)
                        break
                    except Exception as e:
                        last_err = e
                        if attempt == attempts:
                            raise
                        time.sleep(min(20, 2 * attempt))
                if offset and offset % (2 * 1024 ** 3) < chunk:
                    log(f"     {offset / 1e9:.1f} GB / {total / 1e9:.1f} GB")
        if offset == total:
            tmp.replace(dest)
            log(f"   done {dest.name}: {dest.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s")
            return dest
    raise RuntimeError(f"ranged download failed: {last_err}")


def sh(cmd, timeout=600, check=False, env=None):
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        log(f"   exit {r.returncode}: {(r.stderr or r.stdout)[-800:]}")
        if check:
            raise SystemExit(f"command failed: {cmd}")
    return r


def find_one(root, name):
    for p in Path(root).rglob(name):
        return p
    return None


def main():
    publish("start", host=os.uname().nodename, cfg={k: v for k, v in CFG.items() if k != "api_key"})
    t_start = time.time()

    smi = sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader")
    log("GPU: " + smi.stdout.strip().replace("\n", " | "))
    if "Tesla T4" not in smi.stdout:
        publish("failed", step="no-gpu", detail=smi.stdout[:300])
        sys.exit(1)

    # 1. runtime: a prebuilt T4 llama-server attached as a Kaggle dataset
    #    (GitHub release assets intermittently 504, and this image has no nvcc)
    def find_servers():
        cands = []
        for root in ("/kaggle/input", str(WORK)):
            p = Path(root)
            if not p.exists():
                continue
            for cand in p.rglob("llama-server"):
                if cand.is_file() and cand.stat().st_size > 1_000_000:
                    cands.append(cand)
        # newest first
        cands.sort(key=lambda c: c.stat().st_mtime, reverse=True)
        return cands

    cands = find_servers()
    if not cands:
        log("   no mounted llama-server; falling back to GitHub release")
        rel = CFG["llama_release"]
        tar = download_ranged(release_urls(rel, CFG["llama_asset"]), WORK / CFG["llama_asset"], label="llama.cpp CUDA")
        cud = download_ranged(release_urls(rel, CFG["cudart_asset"]), WORK / CFG["cudart_asset"], label="cudart")
        for t in (tar, cud):
            shutil.unpack_archive(str(t), str(WORK))
        cands = find_servers()
    if not cands:
        publish("failed", step="llama-missing")
        sys.exit(1)

    def base_libdirs():
        return sorted({str(p.parent) for root in ("/kaggle/input", str(WORK)) if Path(root).exists()
                       for p in Path(root).rglob("*.so*")})

    def materialize(c):
        # mounted /kaggle/input is read-only and may not carry the exec bit
        if os.access(c, os.X_OK) and (c.stat().st_mode & 0o111):
            return c
        dst = WORK / "bin" / (c.parent.name or "x") / c.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c, dst)
        dst.chmod(0o755)
        return dst

    def try_bins(libdirs):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join(libdirs + [env.get("LD_LIBRARY_PATH", "")])
        for c in cands:
            b = materialize(c)
            v = sh([str(b), "--version"], timeout=120, env=env)
            txt = ((v.stdout or "") + (v.stderr or "")).strip()
            if v.returncode == 0:
                return b, txt, env
            log(f"   {b.name} unusable: {txt[-220:]}")
        return None, "", env

    libdirs = base_libdirs()
    server_bin, version, env = try_bins(libdirs)
    if not server_bin:
        log("   installing nvidia-cuda-runtime-cu12 for libcudart.so.12")
        sh([sys.executable, "-m", "pip", "install", "--quiet", "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"], timeout=900)
        import site
        extra = sorted({str(p) for sp in list(site.getsitepackages()) + [site.getusersitepackages()]
                        for p in [Path(sp) / "nvidia" / "cuda_runtime" / "lib", Path(sp) / "nvidia" / "cublas" / "lib"] if p.exists()})
        libdirs = base_libdirs() + extra
        server_bin, version, env = try_bins(libdirs)
    if not server_bin:
        publish("failed", step="runtime")
        sys.exit(1)
    log(f"   llama-server: {server_bin} ({version[:80]})")
    publish("runtime-ready", version=version[:200])

    # 2. the model
    model_path = WORK / CFG["model_file"]
    if not model_path.exists():
        download(
            f"https://huggingface.co/{CFG['model_repo']}/resolve/main/{CFG['model_file']}",
            model_path, expected=int(CFG["model_bytes"]), label=CFG["model_file"],
        )
    publish("weights-ready", gb=round(model_path.stat().st_size / 1e9, 2), secs=int(time.time() - t_start))

    # 3. serve
    args = [
        str(server_bin),
        "-m", str(model_path),
        "--host", "127.0.0.1", "--port", str(PORT),
        "--n-gpu-layers", "99",
        "--ctx-size", str(CFG["ctx"]),
        "--parallel", str(CFG["parallel"]),
        "--batch-size", "2048", "--ubatch-size", "512",
        "--api-key", CFG["api_key"],
        "--alias", CFG["served_model_name"],
        "--jinja",
        "--no-webui",
        "-t", "4",
    ]
    if CFG.get("tensor_split"):
        args += ["--tensor-split", CFG["tensor_split"]]
    log("launching llama-server")
    logf = open(WORK / "llama-server.log", "w")
    srv = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, env=env)
    publish("server-launch", ctx=CFG["ctx"], parallel=CFG["parallel"])

    import urllib.error
    healthy = False
    t0 = time.time()
    while time.time() - t0 < 1800:
        if srv.poll() is not None:
            tail = (WORK / "llama-server.log").read_text()[-2500:]
            publish("failed", step="server", tail=tail)
            sys.exit(1)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health",
                                         headers={"Authorization": f"Bearer {CFG['api_key']}"})
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    healthy = True
                    break
        except Exception:
            pass
        el = int(time.time() - t0)
        if el and el % 60 < 5:
            log(f"   loading... {el}s")
        time.sleep(3)
    if not healthy:
        publish("failed", step="health-timeout", tail=(WORK / "llama-server.log").read_text()[-2000:])
        sys.exit(1)
    startup = int(time.time() - t0)
    mem = sh("nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader")
    publish("serving", startup_secs=startup, vram=mem.stdout.strip().replace("\n", " | "))

    # 4. self-test
    def chat(messages, max_tokens=64, stream=False):
        body = json.dumps({
            "model": CFG["served_model_name"], "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.0, "stream": stream,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {CFG['api_key']}"})
        return urllib.request.urlopen(req, timeout=900)

    if CFG["self_test"]:
        try:
            t = time.time()
            with chat([{"role": "user", "content": "The capital of France is"}], 16) as r:
                j = json.loads(r.read())
            log("   sanity: " + repr(j["choices"][0]["message"]["content"][:60]))

            t = time.time()
            ttft = None
            gen = 0
            with chat([{"role": "user", "content": "Write a short story about a lighthouse."}], 128, stream=True) as r:
                for raw in r:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break
                    try:
                        o = json.loads(d)
                    except Exception:
                        continue
                    delta = (o.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                    if delta:
                        if ttft is None:
                            ttft = time.time() - t
                        gen += 1
            total = time.time() - t
            tps = (gen - 1) / (total - ttft) if gen > 1 else 0
            publish("benchmark", ttft_s=round(ttft or 0, 2), decode_tok_s=round(tps, 1),
                    chunks=gen, seconds=round(total, 1))
        except Exception as e:
            publish("benchmark-error", err=str(e)[:300])

    # 5. tunnel
    cfd = WORK / "cloudflared"
    if not cfd.exists():
        mounted = None
        for root in ("/kaggle/input",):
            if Path(root).exists():
                for p in Path(root).rglob("cloudflared"):
                    if p.is_file() and p.stat().st_size > 1_000_000:
                        mounted = p
                        break
        if mounted:
            log(f"   cloudflared from dataset: {mounted}")
            shutil.copy2(mounted, cfd)
            cfd.chmod(0o755)
        else:
            download_ranged([
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                "https://gh-proxy.com/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            ], cfd, label="cloudflared")
            cfd.chmod(0o755)
    tun = subprocess.Popen([str(cfd), "tunnel", "--url", f"http://127.0.0.1:{PORT}",
                            "--no-autoupdate", "--protocol", "quic"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    t0 = time.time()
    while time.time() - t0 < 180:
        line = tun.stdout.readline()
        if not line:
            break
        m = pat.search(line)
        if m:
            url = m.group(0)
            break
    if not url:
        publish("tunnel-failed")
        sys.exit(1)
    log(f"   endpoint {url}/v1")
    publish("tunnel-url", endpoint=f"{url}/v1")
    publish("ready", endpoint=f"{url}/v1", api_key=CFG["api_key"], model=CFG["served_model_name"],
            max_model_len=CFG["ctx"], keepalive_min=CFG["keepalive_min"], startup_secs=startup)

    t_serve = time.time()
    while time.time() - t_serve < CFG["keepalive_min"] * 60:
        time.sleep(120)
        if srv.poll() is not None:
            publish("stopped", reason="server-exit")
            sys.exit(1)
        up = int((time.time() - t_serve) / 60)
        if up % 10 < 2:
            publish("heartbeat", up_min=up, endpoint=f"{url}/v1")
    publish("auto-shutdown", served_min=CFG["keepalive_min"])
    srv.terminate()


if __name__ == "__main__":
    main()
