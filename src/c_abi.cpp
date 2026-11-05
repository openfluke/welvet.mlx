#include "welvet_mlx_c.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "prepared.hpp"
#include "welvet_mlx/detect.hpp"
#include "welvet_mlx/dtype.hpp"
#include "welvet_mlx/layers.hpp"
#include "welvet_mlx/sequential.hpp"
#include "welvet_mlx/session.hpp"

using namespace welvet_mlx;

struct wmlx_session {
  Session sess;
  DensePrepared dense_prep;
};

struct wmlx_sequential {
  Sequential net;
};

namespace {

wmlx_status ok() {
  wmlx_status s{};
  s.code = 0;
  s.message[0] = '\0';
  return s;
}

wmlx_status from(const Status& st) {
  wmlx_status s{};
  s.code = (int32_t)st.code;
  std::snprintf(s.message, sizeof(s.message), "%s", st.message.c_str());
  return s;
}

void copy_str(char* dst, int32_t n, const std::string& src) {
  if (!dst || n <= 0) return;
  std::snprintf(dst, (size_t)n, "%s", src.c_str());
}

DType parse_dtype(const char* d) {
  if (!d || !d[0]) return DType::Float32;
  if (std::strcmp(d, "float16") == 0 || std::strcmp(d, "f16") == 0) return DType::Float16;
  if (std::strcmp(d, "bfloat16") == 0 || std::strcmp(d, "bf16") == 0) return DType::BFloat16;
  if (std::strcmp(d, "float64") == 0 || std::strcmp(d, "f64") == 0) return DType::Float64;
  if (std::strcmp(d, "int8") == 0 || std::strcmp(d, "i8") == 0) return DType::Int8;
  if (std::strcmp(d, "int16") == 0 || std::strcmp(d, "i16") == 0) return DType::Int16;
  if (std::strcmp(d, "int32") == 0 || std::strcmp(d, "i32") == 0) return DType::Int32;
  if (std::strcmp(d, "int64") == 0 || std::strcmp(d, "i64") == 0) return DType::Int64;
  if (std::strcmp(d, "uint8") == 0 || std::strcmp(d, "u8") == 0) return DType::UInt8;
  if (std::strcmp(d, "uint16") == 0 || std::strcmp(d, "u16") == 0) return DType::UInt16;
  if (std::strcmp(d, "uint32") == 0 || std::strcmp(d, "u32") == 0) return DType::UInt32;
  if (std::strcmp(d, "uint64") == 0 || std::strcmp(d, "u64") == 0) return DType::UInt64;
  if (std::strcmp(d, "complex64") == 0 || std::strcmp(d, "c64") == 0) return DType::Complex64;
  return DType::Float32;
}

LayerId parse_layer(const char* name) {
  if (!name) return LayerId::Count;
  for (int i = 0; i < (int)LayerId::Count; ++i) {
    if (std::strcmp(name, layer_id_name((LayerId)i)) == 0) return (LayerId)i;
  }
  return LayerId::Count;
}

wmlx_status need_layer(const char* layer, const char* dtype) {
  auto id = parse_layer(layer);
  if (id == LayerId::Count || !layer_implemented(id) ||
      !mlx_supports(id, parse_dtype(dtype))) {
    return from(Status::Unsupported(std::string(layer ? layer : "?") + "/" +
                                    dtype_name(parse_dtype(dtype))));
  }
  return ok();
}

} // namespace

extern "C" {

const char* wmlx_version(void) { return "0.01"; }

int32_t wmlx_has_mlx(void) { return 1; }

int32_t wmlx_layer_supported(const char* layer, const char* dtype) {
  auto id = parse_layer(layer);
  if (id == LayerId::Count || !layer_implemented(id)) return 0;
  return mlx_supports(id, parse_dtype(dtype)) ? 1 : 0;
}

const char* const* wmlx_supported_layers(void) { return supported_layers; }

wmlx_status wmlx_detect(int32_t* mlx_found, int32_t* metal_ok, int32_t* gpu_ok, char* mlx_ver,
                        int32_t mlx_ver_n, char* device_name, int32_t device_name_n) {
  auto d = detect();
  if (mlx_found) *mlx_found = d.mlx_found ? 1 : 0;
  if (metal_ok) *metal_ok = d.metal_available ? 1 : 0;
  if (gpu_ok) *gpu_ok = d.gpu_available ? 1 : 0;
  copy_str(mlx_ver, mlx_ver_n, d.mlx_version);
  copy_str(device_name, device_name_n, d.device_name);
  return ok();
}

wmlx_status wmlx_session_open_gpu(wmlx_session** out) {
  if (!out) return from(Status::Failed("nil out"));
  auto* s = new wmlx_session();
  auto st = Session::open_gpu(&s->sess);
  if (!st) {
    delete s;
    return from(st);
  }
  *out = s;
  return ok();
}

void wmlx_session_close(wmlx_session* s) { delete s; }

int32_t wmlx_session_ready(const wmlx_session* s) {
  return (s && s->sess.ready()) ? 1 : 0;
}

wmlx_status wmlx_dense_prepare_f32(wmlx_session* s, int32_t batch, int32_t in, int32_t out,
                                   const float* W, const float* bias, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("dense", dtype);
    if (st.code) return st;
  }
  if (!W) return from(Status::Failed("nil W"));
  return from(s->dense_prep.prepare(s->sess, batch, in, out, W, bias, parse_dtype(dtype)));
}

wmlx_status wmlx_dense_execute_f32(wmlx_session* s, const float* x, float* y) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  if (!x || !y) return from(Status::Failed("nil x/y"));
  return from(s->dense_prep.execute(s->sess, x, y));
}

wmlx_status wmlx_dense_forward_f32(wmlx_session* s, int32_t batch, int32_t in, int32_t out,
                                   const float* W, const float* bias, const float* x, float* y,
                                   const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("dense", dtype);
    if (st.code) return st;
  }
  auto st = s->dense_prep.prepare(s->sess, batch, in, out, W, bias, parse_dtype(dtype));
  if (!st) return from(st);
  return from(s->dense_prep.execute(s->sess, x, y));
}

wmlx_status wmlx_softmax_forward_f32(wmlx_session* s, int32_t rows, int32_t cols, const float* x,
                                     float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("softmax", dtype);
    if (st.code) return st;
  }
  return from(Softmax::blank(cols, parse_dtype(dtype)).forward(s->sess, rows, x, y));
}

wmlx_status wmlx_rmsnorm_forward_f32(wmlx_session* s, int32_t batch, int32_t seq, int32_t dim,
                                     float eps, const float* gamma, const float* x, float* y,
                                     const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("rmsnorm", dtype);
    if (st.code) return st;
  }
  auto r = RmsNorm::blank(dim, eps, parse_dtype(dtype));
  if (gamma) r.gamma.assign(gamma, gamma + dim);
  return from(r.forward(s->sess, batch, seq, x, y));
}

wmlx_status wmlx_layernorm_forward_f32(wmlx_session* s, int32_t batch, int32_t seq, int32_t dim,
                                       float eps, const float* gamma, const float* beta,
                                       const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("layernorm", dtype);
    if (st.code) return st;
  }
  auto l = LayerNorm::blank(dim, eps, parse_dtype(dtype));
  if (gamma) l.gamma.assign(gamma, gamma + dim);
  if (beta) l.beta.assign(beta, beta + dim);
  return from(l.forward(s->sess, batch, seq, x, y));
}

wmlx_status wmlx_seq_create(wmlx_sequential** out) {
  if (!out) return from(Status::Failed("nil out"));
  *out = new wmlx_sequential();
  return ok();
}

void wmlx_seq_free(wmlx_sequential* seq) { delete seq; }

wmlx_status wmlx_seq_push_dense(wmlx_sequential* seq, int32_t in, int32_t out, const float* W,
                                const float* bias) {
  if (!seq) return from(Status::Failed("nil seq"));
  auto d = Dense::blank(in, out);
  if (W) d.weight.assign(W, W + (size_t)out * (size_t)in);
  if (bias)
    d.bias.assign(bias, bias + out);
  else
    d.bias.assign((size_t)out, 0.f);
  return from(seq->net.push(std::move(d)));
}

wmlx_status wmlx_seq_push_rmsnorm(wmlx_sequential* seq, int32_t dim, float eps,
                                  const float* gamma) {
  if (!seq) return from(Status::Failed("nil seq"));
  auto r = RmsNorm::blank(dim, eps);
  if (gamma) r.gamma.assign(gamma, gamma + dim);
  return from(seq->net.push(std::move(r)));
}

wmlx_status wmlx_seq_push_layernorm(wmlx_sequential* seq, int32_t dim, float eps,
                                    const float* gamma, const float* beta) {
  if (!seq) return from(Status::Failed("nil seq"));
  auto l = LayerNorm::blank(dim, eps);
  if (gamma) l.gamma.assign(gamma, gamma + dim);
  if (beta) l.beta.assign(beta, beta + dim);
  return from(seq->net.push(std::move(l)));
}

wmlx_status wmlx_seq_push_softmax(wmlx_sequential* seq, int32_t cols) {
  if (!seq) return from(Status::Failed("nil seq"));
  return from(seq->net.push(Softmax::blank(cols)));
}

int32_t wmlx_seq_in_features(const wmlx_sequential* seq) {
  return seq ? seq->net.in_features() : 0;
}
int32_t wmlx_seq_out_features(const wmlx_sequential* seq) {
  return seq ? seq->net.out_features() : 0;
}
int32_t wmlx_seq_size(const wmlx_sequential* seq) {
  return seq ? (int32_t)seq->net.size() : 0;
}

wmlx_status wmlx_seq_forward_f32(wmlx_session* s, wmlx_sequential* seq, int32_t batch,
                                 int32_t seq_len, const float* x, float* y) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  if (!seq) return from(Status::Failed("nil seq"));
  return from(seq->net.forward(s->sess, batch, x, y, seq_len));
}

void wmlx_last_phase_ns(const wmlx_session* s, int64_t* create_ns, int64_t* finalize_ns,
                        int64_t* execute_ns) {
  if (!s) {
    if (create_ns) *create_ns = 0;
    if (finalize_ns) *finalize_ns = 0;
    if (execute_ns) *execute_ns = 0;
    return;
  }
  if (create_ns) *create_ns = s->sess.last_phase.create_ns;
  if (finalize_ns) *finalize_ns = s->sess.last_phase.finalize_ns;
  if (execute_ns) *execute_ns = s->sess.last_phase.execute_ns;
}

wmlx_status wmlx_cnn2_forward_f32(wmlx_session* s, int32_t n, int32_t h, int32_t w, int32_t cin,
                                  int32_t cout, const float* W, const float* bias, const float* x,
                                  float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("cnn2", dtype);
    if (st.code) return st;
  }
  auto c = Cnn2::blank(cin, cout, h, w, parse_dtype(dtype));
  c.n = n;
  if (W) c.weight.assign(W, W + (size_t)cout * (size_t)cin);
  if (bias) c.bias.assign(bias, bias + cout);
  return from(c.forward(s->sess, x, y));
}

wmlx_status wmlx_cnn3_forward_f32(wmlx_session* s, int32_t n, int32_t d, int32_t h, int32_t w,
                                  int32_t cin, int32_t cout, const float* W, const float* bias,
                                  const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("cnn3", dtype);
    if (st.code) return st;
  }
  auto c = Cnn3::blank(cin, cout, d, h, w, parse_dtype(dtype));
  c.n = n;
  if (W) c.weight.assign(W, W + (size_t)cout * (size_t)cin);
  if (bias) c.bias.assign(bias, bias + cout);
  return from(c.forward(s->sess, x, y));
}

wmlx_status wmlx_convt2_forward_f32(wmlx_session* s, int32_t n, int32_t h, int32_t w, int32_t cin,
                                    int32_t cout, const float* W, const float* bias, const float* x,
                                    float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("convt2", dtype);
    if (st.code) return st;
  }
  auto c = ConvT2::blank(cin, cout, h, w, parse_dtype(dtype));
  c.n = n;
  if (W) c.weight.assign(W, W + (size_t)cout * (size_t)cin);
  if (bias) c.bias.assign(bias, bias + cout);
  return from(c.forward(s->sess, x, y));
}

wmlx_status wmlx_convt3_forward_f32(wmlx_session* s, int32_t n, int32_t d, int32_t h, int32_t w,
                                    int32_t cin, int32_t cout, const float* W, const float* bias,
                                    const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("convt3", dtype);
    if (st.code) return st;
  }
  auto c = ConvT3::blank(cin, cout, d, h, w, parse_dtype(dtype));
  c.n = n;
  if (W) c.weight.assign(W, W + (size_t)cout * (size_t)cin);
  if (bias) c.bias.assign(bias, bias + cout);
  return from(c.forward(s->sess, x, y));
}

wmlx_status wmlx_lstm_forward_f32(wmlx_session* s, int32_t seq, int32_t batch, int32_t input_size,
                                  int32_t hidden_size, const float* w_in, const float* w_rec,
                                  const float* b, const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("lstm", dtype);
    if (st.code) return st;
  }
  auto l = Lstm::blank(input_size, hidden_size, parse_dtype(dtype));
  if (w_in) l.w_in.assign(w_in, w_in + (size_t)4 * hidden_size * input_size);
  if (w_rec) l.w_rec.assign(w_rec, w_rec + (size_t)4 * hidden_size * hidden_size);
  if (b) l.b.assign(b, b + (size_t)4 * hidden_size);
  return from(l.forward(s->sess, seq, batch, x, y));
}

wmlx_status wmlx_embedding_forward_f32(wmlx_session* s, int32_t vocab, int32_t dim, int32_t n,
                                       const float* table, const int32_t* ids, float* y,
                                       const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("embedding", dtype);
    if (st.code) return st;
  }
  auto e = Embedding::blank(vocab, dim, parse_dtype(dtype));
  if (table) e.table.assign(table, table + (size_t)vocab * dim);
  return from(e.forward(s->sess, n, ids, y));
}

wmlx_status wmlx_residual_forward_f32(wmlx_session* s, int32_t n, const float* a, const float* b,
                                      float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("residual", dtype);
    if (st.code) return st;
  }
  return from(Residual::blank(parse_dtype(dtype)).forward(s->sess, n, a, b, y));
}

wmlx_status wmlx_swiglu_forward_f32(wmlx_session* s, int32_t batch, int32_t in, int32_t hidden,
                                    const float* w_gate, const float* w_up, const float* w_down,
                                    const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("swiglu", dtype);
    if (st.code) return st;
  }
  auto g = SwiGlu::blank(in, hidden, parse_dtype(dtype));
  if (w_gate) g.w_gate.assign(w_gate, w_gate + (size_t)hidden * in);
  if (w_up) g.w_up.assign(w_up, w_up + (size_t)hidden * in);
  if (w_down) g.w_down.assign(w_down, w_down + (size_t)in * hidden);
  return from(g.forward(s->sess, batch, x, y));
}

wmlx_status wmlx_mha_forward_f32(wmlx_session* s, int32_t batch, int32_t seq, int32_t dim,
                                 int32_t heads, const float* w_q, const float* w_k, const float* w_v,
                                 const float* w_o, const float* x, float* y, const char* dtype) {
  if (!s || !s->sess.ready()) return from(Status::Unavailable("no GPU session"));
  {
    auto st = need_layer("mha", dtype);
    if (st.code) return st;
  }
  auto m = Mha::blank(dim, heads, parse_dtype(dtype));
  if (w_q) m.w_q.assign(w_q, w_q + (size_t)dim * dim);
  if (w_k) m.w_k.assign(w_k, w_k + (size_t)dim * dim);
  if (w_v) m.w_v.assign(w_v, w_v + (size_t)dim * dim);
  if (w_o) m.w_o.assign(w_o, w_o + (size_t)dim * dim);
  return from(m.forward(s->sess, batch, seq, x, y));
}

} // extern "C"
