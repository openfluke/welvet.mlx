# welvet.mlx versioning

Forward-only Apple MLX GPU backend. Training is **not** part of this board  
(see `welvet.metal` / `welvet.cuda` for train engines).

| Tag | Meaning |
|-----|---------|
| **0.00** | Empty repo scaffold |
| **0.01** | M0+M1 baseline: MLX GPU session, C ABI, leaf forwards in C++, smoke tool, docs + w2a skeleton |
| **0.x** | `x` = number of **w2a-signed** leaf FormatNone forwards (see [docs/FORWARD_CHECKLIST.md](docs/FORWARD_CHECKLIST.md)) |
| **1.0.0** | Full README leaf set signed in `w2a` `go_v_mlx` for **f32+f16+bf16** (int/c64 optional credits; not Welvet ×34) |

Patch versions (`0.01.y`) = bugfix without new signed leaves.

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
