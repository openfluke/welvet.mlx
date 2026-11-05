#include "prepared.hpp"

#include "util.hpp"

namespace welvet_mlx {

void DensePrepared::clear() {
  W_t.reset();
  bias.reset();
  ready = false;
  batch = in = out = 0;
}

Status DensePrepared::prepare(Session& sess, int32_t batch_, int32_t in_, int32_t out_,
                              const float* W, const float* bias_f, DType dt) {
  auto st = util::require_sess(sess);
  if (!st) return st;
  if (!W || batch_ <= 0 || in_ <= 0 || out_ <= 0) return Status::Failed("dense prepare: bad args");
  if (!mlx_supports(LayerId::Dense, dt))
    return Status::Unsupported(std::string("dense/") + dtype_name(dt));

  try {
    const int64_t t0 = util::mono_ns();
    auto W_host = util::host_f32(W, {out_, in_}); // [out, in]
    auto Wt = util::cast_compute(mlx::core::transpose(W_host, util::gpu()), dt);
    mlx::core::eval(Wt);

    mlx::core::array b =
        bias_f ? util::cast_compute(util::host_f32(bias_f, {out_}), dt)
               : util::cast_compute(mlx::core::zeros({out_}, mlx::core::float32, util::gpu()), dt);
    mlx::core::eval(b);

    W_t = std::move(Wt);
    bias = std::move(b);
    batch = batch_;
    in = in_;
    out = out_;
    dtype = dt;
    ready = true;

    sess.last_phase.create_ns = util::mono_ns() - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = 0;
    return Status::Ok();
  } catch (const std::exception& e) {
    clear();
    return Status::Failed(std::string("dense prepare: ") + e.what());
  }
}

Status DensePrepared::execute(Session& sess, const float* x, float* y) {
  auto st = util::require_sess(sess);
  if (!st) return st;
  if (!ready || !W_t || !bias) return Status::Failed("dense execute: not prepared");
  if (!x || !y) return Status::Failed("dense execute: nil x/y");

  try {
    const int64_t t0 = util::mono_ns();
    auto X = util::cast_compute(util::host_f32(x, {batch, in}), dtype);
    auto Y = mlx::core::add(mlx::core::matmul(X, *W_t, util::gpu()), *bias, util::gpu());
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    const int64_t t2 = util::mono_ns();

    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = t2 - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("dense execute: ") + e.what());
  }
}

} // namespace welvet_mlx
