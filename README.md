# welvet.mlx — Apple MLX GPU (forward-only)

Standalone **C++20** library. Target = **MLX GPU forward** (Metal under libmlx).  
Public surface is **Welvet-shaped** (layer blanks + `DType`) + **C ABI** (`welvet_mlx_c.h`) for Welvet Go `nativemlx`.

| | |
|--|--|
| **Library** | **0.02** |
| **MLX** | Homebrew / `MLX_ROOT` (tested against **0.32.x**) |
| **Host** | macOS · Apple silicon · Metal via MLX |
| **Go** | `nativemlx` · `BackendMLX` · user `SyncToMLX` / `SyncFromMLX` |
| **Bench** | `w2a` `go_v_mlx` |

**No CPU fallback. No silent welvet.metal redirect.** Missing Metal → hard `Unavailable`.

See [VERSIONING.md](VERSIONING.md), [docs/MLX_ONLY.md](docs/MLX_ONLY.md),  
[docs/FORWARD_CHECKLIST.md](docs/FORWARD_CHECKLIST.md).

---

## Layer × dtype (baseline)

**Compute ceiling:** MLX ordinary GPU dtypes that overlap Welvet FormatNone  
(`f32/f16/bf16` + ints + `complex64`). **Not** Welvet ×34 / packed×20.  
Float64 = MLX CPU-only → caps reject on GPU.  
See [docs/FORWARD_CHECKLIST.md](docs/FORWARD_CHECKLIST.md#dtype-ceiling-honest).

Legend: ✅ caps · — rejected · ints = i8…i64 + u8…u64

| Layer | f32 | f16 | bf16 | ints | c64 | Notes |
|-------|-----|-----|------|------|-----|-------|
| Dense | ✅ | ✅ | ✅ | ✅ | ✅ | Prepare-once + execute |
| Softmax | ✅ | ✅ | ✅ | — | — | float math |
| RMSNorm | ✅ | ✅ | ✅ | — | — | `fast::rms_norm` |
| LayerNorm | ✅ | ✅ | ✅ | — | — | `fast::layer_norm` |
| CNN2 | ✅ | ✅ | ✅ | ✅ | — | NHWC |
| CNN3 | ✅ | ✅ | ✅ | ✅ | — | |
| ConvT2 | ✅ | ✅ | ✅ | ✅ | — | |
| ConvT3 | ✅ | ✅ | ✅ | ✅ | — | |
| LSTM | ✅ | ✅ | ✅ | — | — | Gate compose |
| Embedding | ✅ | ✅ | ✅ | ✅ | — | |
| Residual | ✅ | ✅ | ✅ | ✅ | ✅ | |
| SwiGLU | ✅ | ✅ | ✅ | — | — | SiLU = x·σ(x) |
| MHA | ✅ | ✅ | ✅ | — | — | SDPA |

Skips are intentional caps in `mlx_supports()` — not silent fails.

**Not in baseline:** CNN1, ConvT1, RNN, KMeans, Mamba, GDN, Parallel, Stack, metacognition, seqmix — **all training** — exotic/quant storage truth.

---

## Sequential

```cpp
#include "welvet_mlx/sequential.hpp"

welvet_mlx::Session sess;
welvet_mlx::Session::open_gpu(&sess);

welvet_mlx::Sequential net;
net.push(welvet_mlx::Dense::blank(4, 8));
net.push(welvet_mlx::RmsNorm::blank(8));
net.push(welvet_mlx::Dense::blank(8, 4));
net.push(welvet_mlx::Softmax::blank(4));

net.forward(sess, /*batch=*/1, x, y, /*seq=*/1);
```

Each child is its own MLX eval (host ping-pong). Fused Sequential = later.

---

## C ABI → Welvet Go

| Artifact | Location |
|----------|----------|
| Header | `include/welvet_mlx_c.h` → `welvet/native/mlx/include/` |
| dylib | `libwelvet_mlx.dylib` → `welvet/native/mlx/darwin_arm64/` |
| Go | `github.com/openfluke/welvet/nativemlx` |

```text
SyncToMLX → BackendMLX Forward → SyncFromMLX (optional)
```

Timing: **exec** = `mx::eval` window inside this library (`wmlx_last_phase_ns`).

---

## Build / smoke

```sh
brew install mlx          # or set MLX_ROOT
./scripts/build_macos.sh
./build/welvet_mlx_smoke
./scripts/macos_build_w2a.sh --smoke
```

| Target | Role |
|--------|------|
| `libwelvet_mlx` | Shared dylib + C ABI |
| `welvet_mlx_smoke` | Full layer×dtype on GPU |

---

## Next (0.x → 1.0.0)

1. Sign every leaf in `w2a` `go_v_mlx` (checklist)
2. Prepare-once for norms / softmax / cnn
3. Quant / int compute paths (if needed)
4. Optional volume forward (still no train)
