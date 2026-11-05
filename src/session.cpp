#include "welvet_mlx/session.hpp"

#include <string>
#include <variant>

#include <mlx/mlx.h>

namespace welvet_mlx {

Status Session::open_gpu(Session* out) {
  if (!out) return Status::Failed("nil session out");

  if (!mlx::core::metal::is_available()) {
    return Status::Unavailable(
        "Metal not available — welvet.mlx is GPU-only (no CPU fallback)");
  }
  if (!mlx::core::is_available(mlx::core::Device(mlx::core::Device::gpu))) {
    return Status::Unavailable("MLX GPU device unavailable");
  }

  try {
    mlx::core::set_default_device(mlx::core::Device(mlx::core::Device::gpu));
    // Touch the device so load failures surface at open, not first forward.
    auto probe = mlx::core::zeros({1}, mlx::core::float32, mlx::core::Device::gpu);
    mlx::core::eval(probe);

    out->ok_ = true;
    out->mlx_version_ = mlx::core::version();
    auto info = mlx::core::device_info(mlx::core::Device(mlx::core::Device::gpu));
    if (auto it = info.find("device_name"); it != info.end()) {
      if (auto* s = std::get_if<std::string>(&it->second)) out->device_name_ = *s;
    }
    if (out->device_name_.empty()) out->device_name_ = "mlx-gpu";
    return Status::Ok();
  } catch (const std::exception& e) {
    out->ok_ = false;
    return Status::Unavailable(std::string("mlx GPU open failed: ") + e.what());
  }
}

} // namespace welvet_mlx
