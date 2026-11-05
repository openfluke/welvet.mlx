# welvet.mlx gaps

Machine census for forward-only MLX. Update when signing w2a rows.

## Done (1.0.0 — 14 signed leaves, first stable forward ship)

- [x] Detect / Metal+GPU gate (no CPU open path)
- [x] Session::open_gpu
- [x] C ABI (`welvet_mlx_c.h`)
- [x] Dense prepare + execute + one-shot (all Dense caps dtypes)
- [x] Go `SyncToMLX` / `PrepareMLX` / `ForwardMLX` / `BackendMLX` on **all** baseline leaves
- [x] Softmax / RMSNorm / LayerNorm / Sequential mlx.go
- [x] CNN2 / CNN3 / ConvT2 / ConvT3 / LSTM / Embedding / Residual / SwiGLU / MHA mlx.go
- [x] w2a `go_v_mlx` Session×caps dtypes for every leaf + `TestGoVMLXAll`
- [x] `welvet_mlx_smoke`
- [x] Docs: MLX_ONLY + FORWARD_CHECKLIST + VERSIONING

## Open

- [ ] Dense prepare-once warm path timing vs online
- [ ] Fused Sequential single graph (today: child stack eval)
- [ ] Volume / Grid forward (optional later)
- [ ] Ship `libwelvet_mlx.dylib` into `welvet/native/mlx/darwin_arm64` release lane
- [ ] Optional later: mapped MLX quant / custom Metal for exotic Welvet storage (never silent)
- [ ] Paste measured Metal host bands into RESULTS.md

## Never (this repo) / hard reject

- Backward / TrainMode / Tween / Cameral / CamSync
- Silent CPU or Metal-shader fallback
- Claiming Welvet FormatNone×34 or packed×20 as native MLX GPU dtypes
- GPU Float64 (MLX float64 is CPU-only)
- ANE as an MLX device
