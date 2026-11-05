#include "welvet_mlx/dtype.hpp"

namespace welvet_mlx {

bool mlx_supports(LayerId layer, DType dtype) {
  // MLX float64 is CPU-only — never claim GPU.
  if (dtype == DType::Float64) return false;

  const bool fl =
      dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16;
  const bool ix = dtype_is_int(dtype);
  const bool cx = dtype == DType::Complex64;

  if (!fl && !ix && !cx) return false;

  switch (layer) {
    case LayerId::Dense:
      // MLX matmul is float/complex only (Welvet/w2a uses float paths).
      return fl || cx;

    case LayerId::Residual:
      return fl || ix || cx;

    case LayerId::Cnn2:
    case LayerId::Cnn3:
    case LayerId::ConvT2:
    case LayerId::ConvT3:
      // MLX conv is floating-point only.
      return fl;

    case LayerId::Embedding:
      return fl || ix;

    case LayerId::Softmax:
    case LayerId::RmsNorm:
    case LayerId::LayerNorm:
    case LayerId::Lstm:
    case LayerId::SwiGlu:
    case LayerId::Mha:
      // Continuous / fused float math — ints & complex not in baseline caps.
      return fl;

    default:
      return false;
  }
}

bool layer_implemented(LayerId layer) {
  switch (layer) {
    case LayerId::Dense:
    case LayerId::Softmax:
    case LayerId::RmsNorm:
    case LayerId::LayerNorm:
    case LayerId::Cnn2:
    case LayerId::Cnn3:
    case LayerId::ConvT2:
    case LayerId::ConvT3:
    case LayerId::Lstm:
    case LayerId::Embedding:
    case LayerId::Residual:
    case LayerId::SwiGlu:
    case LayerId::Mha:
      return true;
    default:
      return false;
  }
}

} // namespace welvet_mlx
