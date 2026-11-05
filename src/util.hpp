#pragma once

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "welvet_mlx/dtype.hpp"
#include "welvet_mlx/session.hpp"
#include "welvet_mlx/status.hpp"

namespace welvet_mlx {
namespace util {

namespace mx = mlx::core;

inline int64_t mono_ns() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch())
      .count();
}

inline mx::Dtype to_mlx(DType d) {
  switch (d) {
    case DType::Float16:
      return mx::float16;
    case DType::BFloat16:
      return mx::bfloat16;
    case DType::Float64:
      return mx::float64; // CPU-only in MLX — callers must not schedule on GPU
    case DType::Int8:
      return mx::int8;
    case DType::Int16:
      return mx::int16;
    case DType::Int32:
      return mx::int32;
    case DType::Int64:
      return mx::int64;
    case DType::UInt8:
      return mx::uint8;
    case DType::UInt16:
      return mx::uint16;
    case DType::UInt32:
      return mx::uint32;
    case DType::UInt64:
      return mx::uint64;
    case DType::Complex64:
      return mx::complex64;
    case DType::Float32:
    default:
      return mx::float32;
  }
}

inline mx::Device gpu() { return mx::Device(mx::Device::gpu); }

inline mx::array host_f32(const float* p, mx::Shape shape) {
  return mx::array(p, std::move(shape), mx::float32);
}

inline mx::array host_i32(const int32_t* p, mx::Shape shape) {
  return mx::array(p, std::move(shape), mx::int32);
}

inline mx::array cast_compute(mx::array a, DType dt) {
  auto want = to_mlx(dt);
  if (a.dtype() == want) return a;
  return mx::astype(std::move(a), want, gpu());
}

inline void copy_out_f32(mx::array a, float* out) {
  a = mx::astype(std::move(a), mx::float32, gpu());
  mx::eval(a);
  std::memcpy(out, a.data<float>(), a.size() * sizeof(float));
}

inline Status require_sess(Session& sess) {
  if (!sess.ready()) return Status::Unavailable("mlx session not ready (GPU required)");
  return Status::Ok();
}

inline Status catch_mlx(Session& sess, int64_t create_ns, int64_t exec_ns, const char* what,
                        auto&& fn) {
  try {
    fn();
    sess.last_phase.create_ns = create_ns;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = exec_ns;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string(what) + ": " + e.what());
  }
}

} // namespace util
} // namespace welvet_mlx
