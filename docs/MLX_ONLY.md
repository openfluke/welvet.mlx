# MLX execution contract

`welvet.mlx` is a standalone **forward-only** Apple MLX GPU backend.  
Sibling backends stay isolated:

| Repo | Role |
|------|------|
| `welvet` | Go reference |
| `welvet.cpp` | CPU core |
| `welvet.metal` | Apple Metal shaders (full train engine) |
| `welvet.cuda` | NVIDIA CUDA |
| `welvet.qnn` | Qualcomm Hexagon NPU (forward) |
| **welvet.mlx** | **Apple MLX C++ (forward only)** |

There is **no host / CPU numerical fallback** labeled as MLX, and **no silent fallthrough** to `welvet.metal` or Go.

## Supported computation

**M0** (scaffold — board credit when GPU open works):

- MLX detect + Metal/GPU availability gate
- `Session::open_gpu` (hard fail if Metal/GPU missing)
- C ABI: version / detect / session / caps
- Smoke tool `welvet_mlx_smoke`

**M1+** (engine credit — leaf FormatNone forwards on MLX GPU):

- Dense (+ prepare-once / execute)
- Softmax, RMSNorm, LayerNorm
- CNN2/3, ConvT2/3
- LSTM (MLX-composed gates), Embedding, Residual
- SwiGLU, MHA (`fast::scaled_dot_product_attention`)
- Sequential (host ping-pong of child MLX evals)

**Out of scope (this repo):**

- Backward / TrainMode / Tween / Cameral / volume credit
- Anything that would require a host numerical path

See [FORWARD_CHECKLIST.md](FORWARD_CHECKLIST.md).

## Non-negotiable

- Supported numerical forwards run on **MLX GPU** (`Device::gpu` / Metal via libmlx).
- Host code may schedule, validate, allocate and stage float32 storage.
- Unsupported dtype/layer → hard error (`Unsupported` / `Unavailable`).
- Missing Metal → `Unavailable` — **never** run the same math on CPU and call it MLX.
- **Dtype ceiling:** GPU ordinary MLX ∩ Welvet FormatNone —  
  `f32/f16/bf16` (all leaves), `i8…i64` / `u8…u64` (Dense/CNN/ConvT/Emb/Residual),  
  `complex64` (Dense/Residual). See checklist caps table.  
  No silent cast of exotic/quant Welvet storage.  
  MLX `float64` is CPU-only — never claim GPU Float64.
- No ANE device — MLX exposes CPU/GPU only; Neural Engine is out of scope.
- Parity vs Go is best-effort on w2a's spectrum; report the band **per dtype**.

## Build and delivery

```sh
./scripts/build_macos.sh
./build/welvet_mlx_smoke
./scripts/macos_install_go.sh          # → welvet/native/mlx/ + MANIFEST
./scripts/macos_build_w2a.sh           # install + w2a go_v_mlx
./scripts/macos_build_w2a.sh --smoke   # install + C++ smoke only
```

Requires: macOS, CMake ≥ 3.20, C++20, Homebrew `mlx` (or `MLX_ROOT`).

Tests and Go parity live in sibling `w2a` (`go_v_mlx`), not this repository.  
Shared artifact: `libwelvet_mlx.dylib`.
