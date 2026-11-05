#include "welvet_mlx/layers.hpp"

#include <cmath>
#include <cstring>

#include <mlx/mlx.h>

#include "util.hpp"

namespace welvet_mlx {
namespace mx = mlx::core;

Dense Dense::blank(int32_t in, int32_t out, DType dt) {
  Dense d;
  d.in = in;
  d.out = out;
  d.dtype = dt;
  d.weight.assign((size_t)out * (size_t)in, 0.f);
  d.bias.assign((size_t)out, 0.f);
  const int n = in < out ? in : out;
  for (int i = 0; i < n; ++i) d.weight[(size_t)i * (size_t)in + (size_t)i] = 1.f;
  return d;
}

Softmax Softmax::blank(int32_t cols, DType dt) {
  Softmax s;
  s.cols = cols;
  s.dtype = dt;
  return s;
}

RmsNorm RmsNorm::blank(int32_t dim, float eps, DType dt) {
  RmsNorm r;
  r.dim = dim;
  r.eps = eps;
  r.dtype = dt;
  r.gamma.assign((size_t)dim, 1.f);
  return r;
}

LayerNorm LayerNorm::blank(int32_t dim, float eps, DType dt) {
  LayerNorm l;
  l.dim = dim;
  l.eps = eps;
  l.dtype = dt;
  l.gamma.assign((size_t)dim, 1.f);
  l.beta.assign((size_t)dim, 0.f);
  return l;
}

Cnn2 Cnn2::blank(int32_t cin, int32_t cout, int32_t h, int32_t w, DType dt) {
  Cnn2 c;
  c.cin = cin;
  c.cout = cout;
  c.h = h;
  c.w = w;
  c.dtype = dt;
  c.weight.assign((size_t)cout * 1 * 1 * (size_t)cin, 0.f);
  c.bias.assign((size_t)cout, 0.f);
  const int n = cin < cout ? cin : cout;
  for (int i = 0; i < n; ++i) c.weight[(size_t)i * (size_t)cin + (size_t)i] = 1.f;
  return c;
}

Cnn3 Cnn3::blank(int32_t cin, int32_t cout, int32_t d, int32_t h, int32_t w, DType dt) {
  Cnn3 c;
  c.cin = cin;
  c.cout = cout;
  c.d = d;
  c.h = h;
  c.w = w;
  c.dtype = dt;
  c.weight.assign((size_t)cout * 1 * 1 * 1 * (size_t)cin, 0.f);
  c.bias.assign((size_t)cout, 0.f);
  const int n = cin < cout ? cin : cout;
  for (int i = 0; i < n; ++i) c.weight[(size_t)i * (size_t)cin + (size_t)i] = 1.f;
  return c;
}

ConvT2 ConvT2::blank(int32_t cin, int32_t cout, int32_t h, int32_t w, DType dt) {
  ConvT2 c;
  c.cin = cin;
  c.cout = cout;
  c.h = h;
  c.w = w;
  c.dtype = dt;
  c.weight.assign((size_t)cout * 1 * 1 * (size_t)cin, 0.f);
  c.bias.assign((size_t)cout, 0.f);
  const int n = cin < cout ? cin : cout;
  for (int i = 0; i < n; ++i) c.weight[(size_t)i * (size_t)cin + (size_t)i] = 1.f;
  return c;
}

ConvT3 ConvT3::blank(int32_t cin, int32_t cout, int32_t d, int32_t h, int32_t w, DType dt) {
  ConvT3 c;
  c.cin = cin;
  c.cout = cout;
  c.d = d;
  c.h = h;
  c.w = w;
  c.dtype = dt;
  c.weight.assign((size_t)cout * 1 * 1 * 1 * (size_t)cin, 0.f);
  c.bias.assign((size_t)cout, 0.f);
  const int n = cin < cout ? cin : cout;
  for (int i = 0; i < n; ++i) c.weight[(size_t)i * (size_t)cin + (size_t)i] = 1.f;
  return c;
}

Lstm Lstm::blank(int32_t input_size, int32_t hidden_size, DType dt) {
  Lstm l;
  l.input_size = input_size;
  l.hidden_size = hidden_size;
  l.dtype = dt;
  l.w_in.assign((size_t)4 * hidden_size * input_size, 0.f);
  l.w_rec.assign((size_t)4 * hidden_size * hidden_size, 0.f);
  l.b.assign((size_t)4 * hidden_size, 0.f);
  return l;
}

Embedding Embedding::blank(int32_t vocab, int32_t dim, DType dt) {
  Embedding e;
  e.vocab = vocab;
  e.dim = dim;
  e.dtype = dt;
  e.table.assign((size_t)vocab * dim, 0.f);
  for (int i = 0; i < vocab && i < dim; ++i) e.table[(size_t)i * dim + (size_t)i] = 1.f;
  return e;
}

Residual Residual::blank(DType dt) {
  Residual r;
  r.dtype = dt;
  return r;
}

SwiGlu SwiGlu::blank(int32_t in, int32_t hidden, DType dt) {
  SwiGlu s;
  s.in = in;
  s.hidden = hidden;
  s.dtype = dt;
  s.w_gate.assign((size_t)hidden * in, 0.f);
  s.w_up.assign((size_t)hidden * in, 0.f);
  s.w_down.assign((size_t)in * hidden, 0.f);
  const int n = in < hidden ? in : hidden;
  for (int i = 0; i < n; ++i) {
    s.w_gate[(size_t)i * in + i] = 1.f;
    s.w_up[(size_t)i * in + i] = 1.f;
    s.w_down[(size_t)i * hidden + i] = 1.f;
  }
  return s;
}

Mha Mha::blank(int32_t dim, int32_t heads, DType dt) {
  Mha m;
  m.dim = dim;
  m.heads = heads;
  m.dtype = dt;
  m.w_q.assign((size_t)dim * dim, 0.f);
  m.w_k.assign((size_t)dim * dim, 0.f);
  m.w_v.assign((size_t)dim * dim, 0.f);
  m.w_o.assign((size_t)dim * dim, 0.f);
  for (int i = 0; i < dim; ++i) {
    m.w_q[(size_t)i * dim + i] = 1.f;
    m.w_k[(size_t)i * dim + i] = 1.f;
    m.w_v[(size_t)i * dim + i] = 1.f;
    m.w_o[(size_t)i * dim + i] = 1.f;
  }
  return m;
}

static Status need(LayerId id, DType dt) {
  if (!layer_implemented(id)) return Status::Unsupported(layer_id_name(id));
  if (!mlx_supports(id, dt))
    return Status::Unsupported(std::string(layer_id_name(id)) + "/" + dtype_name(dt));
  return Status::Ok();
}

Status Dense::forward(Session& sess, int32_t batch, const float* x, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::Dense, dtype);
  if (!st) return st;
  if (!x || !y || batch <= 0 || in <= 0 || out <= 0) return Status::Failed("dense: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto X = util::cast_compute(util::host_f32(x, {batch, in}), dtype);
    auto W = util::cast_compute(util::host_f32(weight.data(), {out, in}), dtype);
    auto B = util::cast_compute(util::host_f32(bias.data(), {out}), dtype);
    auto Y = mx::add(mx::matmul(X, mx::transpose(W, util::gpu()), util::gpu()), B, util::gpu());
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("dense: ") + e.what());
  }
}

Status Softmax::forward(Session& sess, int32_t rows, const float* x, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::Softmax, dtype);
  if (!st) return st;
  if (!x || !y || rows <= 0 || cols <= 0) return Status::Failed("softmax: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto X = util::cast_compute(util::host_f32(x, {rows, cols}), dtype);
    auto Y = mx::softmax(X, /*axis=*/-1, /*precise=*/true, util::gpu());
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("softmax: ") + e.what());
  }
}

Status RmsNorm::forward(Session& sess, int32_t batch, int32_t seq, const float* x,
                        float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::RmsNorm, dtype);
  if (!st) return st;
  if (!x || !y || batch <= 0 || seq <= 0 || dim <= 0) return Status::Failed("rmsnorm: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto X = util::cast_compute(util::host_f32(x, {batch * seq, dim}), dtype);
    auto G = util::cast_compute(util::host_f32(gamma.data(), {dim}), dtype);
    auto Y = mx::fast::rms_norm(X, G, eps, util::gpu());
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("rmsnorm: ") + e.what());
  }
}

Status LayerNorm::forward(Session& sess, int32_t batch, int32_t seq, const float* x,
                          float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::LayerNorm, dtype);
  if (!st) return st;
  if (!x || !y || batch <= 0 || seq <= 0 || dim <= 0) return Status::Failed("layernorm: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto X = util::cast_compute(util::host_f32(x, {batch * seq, dim}), dtype);
    auto G = util::cast_compute(util::host_f32(gamma.data(), {dim}), dtype);
    auto B = util::cast_compute(util::host_f32(beta.data(), {dim}), dtype);
    auto Y = mx::fast::layer_norm(X, G, B, eps, util::gpu());
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("layernorm: ") + e.what());
  }
}

Status Cnn2::forward(Session& sess, const float* x, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::Cnn2, dtype);
  if (!st) return st;
  if (!x || !y) return Status::Failed("cnn2: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto X = util::cast_compute(util::host_f32(x, {n, h, w, cin}), dtype);
    auto W = util::cast_compute(util::host_f32(weight.data(), {cout, kh, kw, cin}), dtype);
    auto Y = mx::conv2d(X, W, {stride_h, stride_w}, {pad_h, pad_w}, {1, 1}, 1, util::gpu());
    if (!bias.empty()) {
      auto B = util::cast_compute(util::host_f32(bias.data(), {cout}), dtype);
      Y = mx::add(Y, B, util::gpu());
    }
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("cnn2: ") + e.what());
  }
}

Status Cnn3::forward(Session& sess, const float* x, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::Cnn3, dtype);
  if (!st) return st;
  if (!x || !y) return Status::Failed("cnn3: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto X = util::cast_compute(util::host_f32(x, {n, d, h, w, cin}), dtype);
    auto W = util::cast_compute(util::host_f32(weight.data(), {cout, kd, kh, kw, cin}), dtype);
    auto Y = mx::conv3d(X, W, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, 1, util::gpu());
    if (!bias.empty()) {
      auto B = util::cast_compute(util::host_f32(bias.data(), {cout}), dtype);
      Y = mx::add(Y, B, util::gpu());
    }
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("cnn3: ") + e.what());
  }
}

Status ConvT2::forward(Session& sess, const float* x, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::ConvT2, dtype);
  if (!st) return st;
  if (!x || !y) return Status::Failed("convt2: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    // MLX conv_transpose2d weight: [out_channels, kH, kW, in_channels] for some builds;
    // use [cin, kh, kw, cout] layout expected by MLX transpose conv — blank stores [cout,kh,kw,cin].
    // Remap: treat stored Welvet [cout,kh,kw,cin] as MLX weight via transpose axes.
    auto W_stored = util::cast_compute(util::host_f32(weight.data(), {cout, kh, kw, cin}), dtype);
    auto W = mx::transpose(W_stored, {3, 1, 2, 0}, util::gpu()); // → [cin,kh,kw,cout]
    auto X = util::cast_compute(util::host_f32(x, {n, h, w, cin}), dtype);
    auto Y = mx::conv_transpose2d(X, W, {1, 1}, {0, 0}, {1, 1}, {0, 0}, 1, util::gpu());
    if (!bias.empty()) {
      auto B = util::cast_compute(util::host_f32(bias.data(), {cout}), dtype);
      Y = mx::add(Y, B, util::gpu());
    }
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("convt2: ") + e.what());
  }
}

Status ConvT3::forward(Session& sess, const float* x, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::ConvT3, dtype);
  if (!st) return st;
  if (!x || !y) return Status::Failed("convt3: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto W_stored =
        util::cast_compute(util::host_f32(weight.data(), {cout, kd, kh, kw, cin}), dtype);
    auto W = mx::transpose(W_stored, {4, 1, 2, 3, 0}, util::gpu()); // → [cin,kd,kh,kw,cout]
    auto X = util::cast_compute(util::host_f32(x, {n, d, h, w, cin}), dtype);
    auto Y = mx::conv_transpose3d(X, W, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, 1, util::gpu());
    if (!bias.empty()) {
      auto B = util::cast_compute(util::host_f32(bias.data(), {cout}), dtype);
      Y = mx::add(Y, B, util::gpu());
    }
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("convt3: ") + e.what());
  }
}

Status Lstm::forward(Session& sess, int32_t seq, int32_t batch, const float* x, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::Lstm, dtype);
  if (!st) return st;
  if (!x || !y || seq <= 0 || batch <= 0) return Status::Failed("lstm: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    const int H = hidden_size;
    const int I = input_size;
    auto Xin = util::cast_compute(util::host_f32(x, {seq, batch, I}), dtype);
    auto Win = util::cast_compute(util::host_f32(w_in.data(), {4 * H, I}), dtype);
    auto Wrec = util::cast_compute(util::host_f32(w_rec.data(), {4 * H, H}), dtype);
    auto Bb = util::cast_compute(util::host_f32(b.data(), {4 * H}), dtype);

    auto h = util::cast_compute(mx::zeros({batch, H}, mx::float32, util::gpu()), dtype);
    auto c = util::cast_compute(mx::zeros({batch, H}, mx::float32, util::gpu()), dtype);
    std::vector<mx::array> outs;
    outs.reserve((size_t)seq);

    for (int t = 0; t < seq; ++t) {
      auto xt = mx::take(Xin, t, /*axis=*/0, util::gpu()); // [batch, I]
      auto gates = mx::add(
          mx::add(mx::matmul(xt, mx::transpose(Win, util::gpu()), util::gpu()),
                  mx::matmul(h, mx::transpose(Wrec, util::gpu()), util::gpu()), util::gpu()),
          Bb, util::gpu());
      auto parts = mx::split(gates, 4, /*axis=*/-1);
      auto i = mx::sigmoid(parts[0], util::gpu());
      auto f = mx::sigmoid(parts[1], util::gpu());
      auto g = mx::tanh(parts[2], util::gpu());
      auto o = mx::sigmoid(parts[3], util::gpu());
      c = mx::add(mx::multiply(f, c, util::gpu()), mx::multiply(i, g, util::gpu()), util::gpu());
      h = mx::multiply(o, mx::tanh(c, util::gpu()), util::gpu());
      outs.push_back(h);
    }
    auto Y = mx::stack(outs, /*axis=*/0, util::gpu()); // [seq, batch, H]
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("lstm: ") + e.what());
  }
}

Status Embedding::forward(Session& sess, int32_t n, const int32_t* ids, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::Embedding, dtype);
  if (!st) return st;
  if (!ids || !y || n <= 0 || vocab <= 0 || dim <= 0) return Status::Failed("embedding: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto T = util::cast_compute(util::host_f32(table.data(), {vocab, dim}), dtype);
    auto Idx = util::host_i32(ids, {n});
    auto Y = mx::take(T, Idx, /*axis=*/0, util::gpu());
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("embedding: ") + e.what());
  }
}

Status Residual::forward(Session& sess, int32_t n, const float* a, const float* b, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::Residual, dtype);
  if (!st) return st;
  if (!a || !b || !y || n <= 0) return Status::Failed("residual: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto A = util::cast_compute(util::host_f32(a, {n}), dtype);
    auto B = util::cast_compute(util::host_f32(b, {n}), dtype);
    auto Y = mx::add(A, B, util::gpu());
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("residual: ") + e.what());
  }
}

Status SwiGlu::forward(Session& sess, int32_t batch, const float* x, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::SwiGlu, dtype);
  if (!st) return st;
  if (!x || !y || batch <= 0 || in <= 0 || hidden <= 0) return Status::Failed("swiglu: bad args");

  try {
    const int64_t t0 = util::mono_ns();
    auto X = util::cast_compute(util::host_f32(x, {batch, in}), dtype);
    auto Wg = util::cast_compute(util::host_f32(w_gate.data(), {hidden, in}), dtype);
    auto Wu = util::cast_compute(util::host_f32(w_up.data(), {hidden, in}), dtype);
    auto Wd = util::cast_compute(util::host_f32(w_down.data(), {in, hidden}), dtype);
    auto gate = mx::matmul(X, mx::transpose(Wg, util::gpu()), util::gpu());
    auto up = mx::matmul(X, mx::transpose(Wu, util::gpu()), util::gpu());
    // SiLU(gate) * up  (MLX-only: x * sigmoid(x))
    auto silu = mx::multiply(gate, mx::sigmoid(gate, util::gpu()), util::gpu());
    auto hid = mx::multiply(silu, up, util::gpu());
    auto Y = mx::matmul(hid, mx::transpose(Wd, util::gpu()), util::gpu());
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("swiglu: ") + e.what());
  }
}

Status Mha::forward(Session& sess, int32_t batch, int32_t seq, const float* x, float* y) const {
  auto st = util::require_sess(sess);
  if (!st) return st;
  st = need(LayerId::Mha, dtype);
  if (!st) return st;
  if (!x || !y || batch <= 0 || seq <= 0 || dim <= 0 || heads <= 0)
    return Status::Failed("mha: bad args");
  if (dim % heads != 0) return Status::Failed("mha: dim % heads != 0");

  try {
    const int64_t t0 = util::mono_ns();
    const int hd = dim / heads;
    auto X = util::cast_compute(util::host_f32(x, {batch, seq, dim}), dtype);
    auto Wq = util::cast_compute(util::host_f32(w_q.data(), {dim, dim}), dtype);
    auto Wk = util::cast_compute(util::host_f32(w_k.data(), {dim, dim}), dtype);
    auto Wv = util::cast_compute(util::host_f32(w_v.data(), {dim, dim}), dtype);
    auto Wo = util::cast_compute(util::host_f32(w_o.data(), {dim, dim}), dtype);

    auto flat = mx::reshape(X, {batch * seq, dim}, util::gpu());
    auto q = mx::reshape(mx::matmul(flat, mx::transpose(Wq, util::gpu()), util::gpu()),
                         {batch, seq, heads, hd}, util::gpu());
    auto k = mx::reshape(mx::matmul(flat, mx::transpose(Wk, util::gpu()), util::gpu()),
                         {batch, seq, heads, hd}, util::gpu());
    auto v = mx::reshape(mx::matmul(flat, mx::transpose(Wv, util::gpu()), util::gpu()),
                         {batch, seq, heads, hd}, util::gpu());
    // → [B, H, S, Hd] for SDPA
    q = mx::transpose(q, {0, 2, 1, 3}, util::gpu());
    k = mx::transpose(k, {0, 2, 1, 3}, util::gpu());
    v = mx::transpose(v, {0, 2, 1, 3}, util::gpu());
    const float scale = 1.f / std::sqrt((float)hd);
    auto attn = mx::fast::scaled_dot_product_attention(q, k, v, scale, "", std::nullopt,
                                                       std::nullopt, false, util::gpu());
    attn = mx::transpose(attn, {0, 2, 1, 3}, util::gpu()); // [B,S,H,Hd]
    auto merged = mx::reshape(attn, {batch * seq, dim}, util::gpu());
    auto Y = mx::reshape(mx::matmul(merged, mx::transpose(Wo, util::gpu()), util::gpu()),
                         {batch, seq, dim}, util::gpu());
    const int64_t t1 = util::mono_ns();
    util::copy_out_f32(std::move(Y), y);
    sess.last_phase.create_ns = t1 - t0;
    sess.last_phase.finalize_ns = 0;
    sess.last_phase.execute_ns = util::mono_ns() - t1;
    return Status::Ok();
  } catch (const std::exception& e) {
    return Status::Failed(std::string("mha: ") + e.what());
  }
}

} // namespace welvet_mlx
