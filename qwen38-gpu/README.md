# Qwen3.8-27B on a free Kaggle GPU (2× Tesla T4)

**The fallback recipe for accounts Kaggle has not granted a TPU.** It serves the
same model, through the same OpenAI-compatible contract, on the free GPU quota
(30 GPU-hours per account per week) instead of the 20 TPU-hours.

This is **not** a replacement for the TPU recipe: it is ~10× slower per stream
and runs a 4-bit GGUF instead of bf16. It exists because a GPU is handed out to
accounts that are not TPU-verified, and because it boots in a fraction of the
time.

## Measured (Kaggle, 2× Tesla T4 15.36 GB, sm_75)

| What | Measured |
|---|---|
| Weights | `unsloth/Qwen3.8-27B-GGUF`, `UD-Q4_K_M` — 16.46 GB |
| Runtime | prebuilt `llama-server` b12202-class, CUDA, T4 (sm_75) |
| Time to live endpoint | **7 min 23 s** (2 min download, 84 s load, tunnel) |
| Decode, one stream | **13.5–13.7 tok/s** |
| Time to first token | **3.3 s** |
| VRAM | **8.3 GB + 9.4 GB** of 15.36 GB each |
| Context | 32,768 total, 16,384 per slot with `--parallel 2` |
| Cold start from a warm dataset cache | ~5 min |
| Shutdown | keepalive / gateway idle-stop |

## Why llama.cpp and not vLLM

The T4 is **Turing (sm_75)** and the model is **bf16**. Turing has no native
BF16 tensor cores, and vLLM's GPU paths (FlashAttention, FLA/Triton kernels for
the hybrid gated-DeltaNet layers) assume Ampere or newer. llama.cpp runs GGUF
quantized weights in fp16 on Turing and has native `LLM_ARCH_QWEN35` support
(the architecture Qwen3.8 uses). Its `llama-server` speaks the OpenAI API with
SSE streaming and an `--api-key`.

Weights stay quantized (Q4_K_M), so 27B fits in 30 GB of VRAM with room for
context.

## What the kernel does

`kernel/serve_qwen38_gpu.py`:

1. checks for a T4 (`nvidia-smi`), fails fast with `step=no-gpu` otherwise;
2. finds a prebuilt `llama-server` in the attached Kaggle datasets (read-only,
   so it copies the binary to `/tmp` and chmods it); if none, falls back to the
   GitHub release with ranged, retrying downloads;
3. installs `nvidia-cuda-runtime-cu12` + `nvidia-cublas-cu12` if the mounted
   build needs `libcudart.so.12` / `libcublas.so.12` (the Kaggle image has
   torch's CUDA 13 libs, not 12);
4. downloads the GGUF from Hugging Face, retrying on short reads;
5. starts `llama-server` with `--n-gpu-layers 99`, `--jinja`, `--api-key`,
   `--alias qwen3.8-27b`;
6. runs a streaming self-test, publishes the decode speed;
7. opens cloudflared (taken from the TPU env dataset) and publishes
   `ready` with the endpoint and key on the ntfy topic.

## Use it

```bash
# one-shot, from the terminal (needs the Kaggle CLI)
python launch.py serve --model qwen38-27b-gpu

# or through the on-demand gateway
# gateway/config.local.json:  "model": "qwen38-27b-gpu"
node gateway/server.mjs
curl -s localhost:9790/v1/admin/status -H "Authorization: Bearer $KEY" | jq .compute
```

The kernel is also runnable by hand: attach the datasets below, set
`Tesla T4 ×2`, internet on, and Run.

| Dataset | Why |
|---|---|
| `alessandrogemignani/llama-server-t4-cuda` | newest T4 llama.cpp build (CUDA 12.8, libs) |
| `nixty2/llama-cpp-turing-sm75` | monolithic T4 build, fallback |
| `rahim3/qwen38-tpu-env-v5e8` | carries the `cloudflared` binary |

## Limits

- **30 GPU-hours per week per account**, one session at a time. Accounts that
  are not phone-verified cannot use a GPU at all (Kaggle runs them on CPU with
  no internet).
- No vision tower and no MTP speculative decoding in this recipe: text only.
- Quantized weights: Q4_K_M, not the bf16 model the TPU recipe serves.
- ~6 minutes from request to first token when cold; the gateway's idle stop
  keeps it from burning quota between sessions.

## License

MIT for the recipe. `Qwen3.8-27B` weights follow their upstream license; the
GGUF quantization is Unsloth's.
