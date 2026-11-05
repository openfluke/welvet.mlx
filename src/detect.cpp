#include "welvet_mlx/detect.hpp"

#include <string>
#include <variant>

#include <mlx/mlx.h>

namespace welvet_mlx {

DetectResult detect() {
  DetectResult d;
  d.mlx_found = true;
  d.mlx_version = mlx::core::version();
  d.metal_available = mlx::core::metal::is_available();
  d.gpu_available = mlx::core::is_available(mlx::core::Device(mlx::core::Device::gpu));

  if (d.metal_available || d.gpu_available) {
    try {
      auto info = mlx::core::device_info(mlx::core::Device(mlx::core::Device::gpu));
      if (auto it = info.find("device_name"); it != info.end()) {
        if (auto* s = std::get_if<std::string>(&it->second)) d.device_name = *s;
      }
      if (d.device_name.empty()) d.device_name = "mlx-gpu";
      d.note = "MLX GPU ready";
    } catch (const std::exception& e) {
      d.note = e.what();
      d.hints.push_back("device_info failed — Metal may be blocked in this session");
    }
  } else {
    d.note = "MLX GPU / Metal unavailable";
    d.hints.push_back("welvet.mlx requires Apple Metal via libmlx — no CPU fallback");
    d.hints.push_back("brew install mlx  (or build Apple MLX C++)");
  }
  return d;
}

} // namespace welvet_mlx
