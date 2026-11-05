#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <mlx/mlx.h>

#include "welvet_mlx/dtype.hpp"
#include "welvet_mlx/session.hpp"
#include "welvet_mlx/status.hpp"

namespace welvet_mlx {

// Dense prepare-once: bake W/bias on GPU; execute is matmul + eval only.
struct DensePrepared {
  int32_t batch = 0, in = 0, out = 0;
  DType dtype = DType::Float32;
  std::optional<mlx::core::array> W_t; // [in, out] compute dtype
  std::optional<mlx::core::array> bias;
  bool ready = false;

  Status prepare(Session& sess, int32_t batch, int32_t in, int32_t out, const float* W,
                 const float* bias_f, DType dt);
  Status execute(Session& sess, const float* x, float* y);
  void clear();
};

} // namespace welvet_mlx
