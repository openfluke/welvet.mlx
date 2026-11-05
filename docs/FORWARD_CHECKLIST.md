# Forward checklist — welvet.mlx

Forward-only mirror of the leaf-forward section in  
[`welvet.cuda/docs/TRAINING_CHECKLIST.md`](../../welvet.cuda/docs/TRAINING_CHECKLIST.md).  
**Training modes are out of scope** for this repo (use `welvet.metal` / `welvet.cuda` / Go).

Canonical validation: sibling **w2a** `go_v_mlx` (same protocol as `go_v_qnn`:  
transfer → create → exec → e2e + Go accuracy).

**Current engine:** `0.01` — M0 GPU session + C ABI + leaf MLX forwards implemented in-library.  
w2a sign-off rows below start unchecked until `go_v_mlx` evidence lands.

---

## MLX-only mandate (non-negotiable)

**welvet.mlx = MLX GPU compute only.** Host/CPU math belongs in **welvet.cpp** (and Go).  
`welvet.metal` is a separate Metal-shader engine — do not silently redirect.

Unsupported dtype/layer → hard error. Missing Metal → `Unavailable`.

Contract: [MLX_ONLY.md](MLX_ONLY.md).

```sh
./scripts/build_macos.sh
./build/welvet_mlx_smoke
./scripts/macos_build_w2a.sh          # ships dylib + go test ./tests/go_v_mlx
```

---

## Dtype ceiling (honest)

MLX ordinary dtypes ∩ Welvet FormatNone — **not** Welvet’s full ×34 + packed×20.

Host I/O stays **float32**; `dtype` selects the MLX compute cast on GPU.  
Enforced in `mlx_supports()` — no silent cast of exotic/quant storage.

### Caps matrix (library)

| Welvet / name | MLX | Dense / Residual | CNN* / ConvT* / Emb | Softmax / norms / LSTM / SwiGLU / MHA |
|---------------|-----|:----------------:|:-------------------:|:-------------------------------------:|
| Float32 | `float32` | ✅ | ✅ | ✅ |
| Float16 | `float16` | ✅ | ✅ | ✅ |
| BFloat16 | `bfloat16` | ✅ | ✅ | ✅ |
| Float64 | `float64` | — GPU | — GPU | — GPU (MLX CPU-only) |
| Int8 / Int16 / Int32 / Int64 | `int*` | ✅ | ✅ | — |
| UInt8 / UInt16 / UInt32 / UInt64 | `uint*` | ✅ | ✅ | — |
| Complex64 | `complex64` | ✅ | — | — |

### Still out (hard reject — no ordinary MLX dtype / different packing)

FP8E5M2, FP6, NF4, ternary, binary, odd-width ints (i3/i5/…), Welvet packed×20, Complex128, ANE device.

TF32 / Winograd note (parity tests): `MLX_ENABLE_TF32=0`, `MLX_CONV_WINOGRAD=0`.

---

## w2a Go↔MLX (leaf FormatNone forward) — sign-off

Protocol matches `go_v_qnn`: identical Go weights → SyncToMLX → ForwardMLX → accuracy.

**Fully signed off** for a float leaf = f32 **and** f16 **and** bf16 under the declared band.  
Int / complex columns are separate optional credits (tick when evidenced).

| Layer | Test (planned) | f32 | f16 | bf16 | i8…i64 / u* | c64 | Fully signed (floats) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Dense | `TestGoVMLXDenseForward` | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Softmax | `TestGoVMLXSoftmax` | [ ] | [ ] | [ ] | — | — | [ ] |
| RMSNorm | `TestGoVMLXRMSNorm` | [ ] | [ ] | [ ] | — | — | [ ] |
| LayerNorm | `TestGoVMLXLayerNorm` | [ ] | [ ] | [ ] | — | — | [ ] |
| CNN2 | `TestGoVMLXCNN2` | [ ] | [ ] | [ ] | [ ] | — | [ ] |
| CNN3 | `TestGoVMLXCNN3` | [ ] | [ ] | [ ] | [ ] | — | [ ] |
| ConvT2 | `TestGoVMLXConvT2` | [ ] | [ ] | [ ] | [ ] | — | [ ] |
| ConvT3 | `TestGoVMLXConvT3` | [ ] | [ ] | [ ] | [ ] | — | [ ] |
| LSTM | `TestGoVMLXLSTM` | [ ] | [ ] | [ ] | — | — | [ ] |
| Embedding | `TestGoVMLXEmbedding` | [ ] | [ ] | [ ] | [ ] | — | [ ] |
| Residual | `TestGoVMLXResidual` | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| SwiGLU | `TestGoVMLXSwiGLU` | [ ] | [ ] | [ ] | — | — | [ ] |
| MHA | `TestGoVMLXMHA` | [ ] | [ ] | [ ] | — | — | [ ] |
| Sequential | `TestGoVMLXSequential` | [ ] | — | — | — | — | [ ] |

Sequential baseline = **f32** tower only.

### Local smoke (in-repo, not w2a credit)

`welvet_mlx_smoke` sweeps every caps dtype per layer (Float64 rows should **SKIP**/Unsupported).

### Not in scope (engine / layers)

| Item | Status |
|---|---|
| Backward / NormalBP / TrainMode 1–29 | ❌ out of scope |
| Tween / Cameral / CamSync | ❌ out of scope |
| Volume / Grid hop_order | ❌ later |
| CNN1 / ConvT1 / RNN / Mamba / GDN / KMeans / Parallel / Stack / metacognition / seqmix | ❌ not in baseline leaf set |
| Welvet packed quant ×20 as storage truth | ❌ later — never silent |
| GPU Float64 / ANE | ❌ |

### Numerical policy

Parity vs Go: best-effort 60–100% on w2a spectrum. Document bands **per dtype** before signing ✅.  
Int paths round-trip via host f32 cast — expect looser bands than native float leaves.

### Version bumps

Tick a w2a **Fully signed (floats)** row → bump `VERSIONING.md` + `wmlx_version` + CMake `VERSION`.  
Scoreboard: `version = 0.{signed_leaf_count}` → **1.0.0** = float matrix signed (int/c64 credits optional extras).
