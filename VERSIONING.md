# welvet.mlx versioning

Forward-only Apple MLX GPU backend (`libmlx` only). Training is **not** part of this board  
(see `welvet.metal` / `welvet.cuda` for train engines).

| Tag | Meaning |
|-----|---------|
| **0.00** | Empty repo scaffold |
| **0.01** | M0+M1 baseline: session, C ABI, leaf forwards, smoke, docs |
| **0.02** | Dense signed |
| **0.15** | 14 leaves + Session×caps matrix |
| **1.0.0** | First stable forward ship — C ABI + 14-leaf × caps dtype matrix (f32/f16/bf16 + int/c64 where capped). Same bar as welvet.qnn 1.0.0 for “vendor forward subset” |
| **1.x.y** | Apple MLX / macOS only (this repo). Patch = bugfix; minor = new leaves/dtypes/fused Sequential / CNN full kernels |

## Known 1.0.0 gaps (honest)

- Sequential = host ping-pong of child MLX evals (fused single graph = later)
- C ABI `seq_push_*` is **f32-only** today (C++ Sequential can mix child dtypes)
- CNN*/ConvT* C ABI / Go path oriented around **1×1** kernels
- Int/c64 = exec credit; float leaves are the parity target
- No Metal-host RESULTS.md bands attached in-tree yet (run `w2a` `go_v_mlx` on device)
- Go `nativemlx` / `BackendMLX` lives in sibling **welvet** — not this tree
- Not in baseline: CNN1, ConvT1, RNN, Mamba, GDN, KMeans, Parallel, Stack, train, Welvet ×34/quant

## Sibling vendor repos (do not merge into this tree)

| Repo | Vendor / role | OS |
|------|---------------|-----|
| **welvet.mlx** (this) | Apple MLX C++ (forward) | macOS |
| welvet.metal | Apple Metal shaders (full train) | macOS |
| welvet.cuda | NVIDIA CUDA | Windows + Linux |
| welvet.qnn | Qualcomm Hexagon NPU (forward) | Windows ARM64 → Linux ARM64 |
| welvet.cpp | CPU core + C ABI | all |
| welvet | Go reference | all |
| w2a | Cross-backend benches / accuracy | all |

Go binds via thin `nativemlx` (`BackendMLX`).
