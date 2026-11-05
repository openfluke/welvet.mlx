# welvet.mlx gaps

Machine census for forward-only MLX. Update when signing w2a rows.

## Done (0.01 library)

- [x] Detect / Metal+GPU gate (no CPU open path)
- [x] Session::open_gpu
- [x] C ABI (`welvet_mlx_c.h`)
- [x] Dense prepare + execute + one-shot (all Dense caps dtypes)
- [x] Go `dense.SyncToMLX` / `PrepareMLX` / `ForwardMLX` / `BackendMLX`
- [x] w2a Dense suite (caps / Session×dtypes / Go layer floats)
- [x] Softmax / RMSNorm / LayerNorm
- [x] CNN2 / CNN3 / ConvT2 / ConvT3
- [x] LSTM (composed) / Embedding / Residual
- [x] SwiGLU / MHA
- [x] Sequential (unfused child evals)
- [x] `welvet_mlx_smoke`
- [x] Docs: MLX_ONLY + FORWARD_CHECKLIST + VERSIONING

## Open

- [ ] w2a `go_v_mlx` accuracy sign-off (float matrix + optional int/c64 credits)
- [ ] Dense prepare-once warm path timing vs online
- [ ] Fused Sequential single graph
- [ ] Volume / Grid forward (optional later)
- [ ] Ship `libwelvet_mlx.dylib` into `welvet/native/mlx/darwin_arm64` release lane
- [ ] Optional later: mapped MLX quant / custom Metal for exotic Welvet storage (never silent)

## Never (this repo) / hard reject

- Backward / TrainMode / Tween / Cameral / CamSync
- Silent CPU or Metal-shader fallback
- Claiming Welvet FormatNone×34 or packed×20 as native MLX GPU dtypes
- GPU Float64 (MLX float64 is CPU-only)
- ANE as an MLX device

