#include <cstdio>
#include <string>
#include <vector>

#include "welvet_mlx/detect.hpp"
#include "welvet_mlx/dtype.hpp"
#include "welvet_mlx/layers.hpp"
#include "welvet_mlx/sequential.hpp"
#include "welvet_mlx/session.hpp"

using namespace welvet_mlx;

static int g_pass = 0;
static int g_fail = 0;
static int g_skip = 0;

static void ok(const char* name) {
  std::printf("  PASS  %s\n", name);
  ++g_pass;
}
static void fail(const char* name, const std::string& why) {
  std::printf("  FAIL  %s — %s\n", name, why.c_str());
  ++g_fail;
}
static void skip(const char* name, const std::string& why) {
  std::printf("  SKIP  %s — %s\n", name, why.c_str());
  ++g_skip;
}

static void print_detect(const DetectResult& d) {
  std::printf("\n=== detect ===\n");
  std::printf("mlx_found=%d metal=%d gpu=%d ver=%s\n", (int)d.mlx_found, (int)d.metal_available,
              (int)d.gpu_available, d.mlx_version.c_str());
  std::printf("device=%s\n", d.device_name.c_str());
  std::printf("note=%s\n", d.note.c_str());
  for (const auto& h : d.hints) std::printf("  hint: %s\n", h.c_str());
}

static void print_caps() {
  std::printf("\n=== MLX caps (welvet.mlx) ===\n");
  std::printf("%-12s", "layer");
  for (DType dt : kAllDtypes) std::printf(" %9s", dtype_name(dt));
  std::printf("\n");
  for (int i = 0; i < (int)LayerId::Count; ++i) {
    auto id = (LayerId)i;
    std::printf("%-12s", layer_id_name(id));
    for (DType dt : kAllDtypes) {
      if (!mlx_supports(id, dt))
        std::printf(" %9s", "—");
      else
        std::printf(" %9s", layer_implemented(id) ? "api" : "gap");
    }
    std::printf("\n");
  }
}

static void check(const char* name, const Status& st) {
  if (!st) {
    if (st.code == Code::Unsupported)
      skip(name, st.message);
    else
      fail(name, st.message);
  } else {
    ok(name);
  }
}

static void run_one_dtype(Session& sess, LayerId id, DType dt) {
  char name[80];
  std::snprintf(name, sizeof(name), "%s/%s", layer_id_name(id), dtype_name(dt));
  if (!mlx_supports(id, dt)) {
    skip(name, "caps");
    return;
  }
  Status st = Status::Failed("unhandled");
  switch (id) {
    case LayerId::Dense: {
      auto d = Dense::blank(4, 4, dt);
      std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
      st = d.forward(sess, 1, x.data(), y.data());
      break;
    }
    case LayerId::Softmax: {
      auto s = Softmax::blank(4, dt);
      std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
      st = s.forward(sess, 1, x.data(), y.data());
      break;
    }
    case LayerId::RmsNorm: {
      auto r = RmsNorm::blank(4, 1e-5f, dt);
      std::vector<float> x = {1, 1, 1, 1}, y(4, 0);
      st = r.forward(sess, 1, 1, x.data(), y.data());
      break;
    }
    case LayerId::LayerNorm: {
      auto l = LayerNorm::blank(4, 1e-5f, dt);
      std::vector<float> x = {1, 2, 3, 4}, y(4, 0);
      st = l.forward(sess, 1, 1, x.data(), y.data());
      break;
    }
    case LayerId::Cnn2: {
      auto c = Cnn2::blank(2, 2, 4, 4, dt);
      std::vector<float> x(1 * 4 * 4 * 2, 1.f), y(1 * 4 * 4 * 2, 0.f);
      st = c.forward(sess, x.data(), y.data());
      break;
    }
    case LayerId::Cnn3: {
      auto c = Cnn3::blank(2, 2, 2, 2, 2, dt);
      std::vector<float> x(1 * 2 * 2 * 2 * 2, 1.f), y(1 * 2 * 2 * 2 * 2, 0.f);
      st = c.forward(sess, x.data(), y.data());
      break;
    }
    case LayerId::ConvT2: {
      auto c = ConvT2::blank(2, 2, 2, 2, dt);
      std::vector<float> x(1 * 2 * 2 * 2, 1.f), y(1 * 2 * 2 * 2, 0.f);
      st = c.forward(sess, x.data(), y.data());
      break;
    }
    case LayerId::ConvT3: {
      auto c = ConvT3::blank(2, 2, 2, 2, 2, dt);
      std::vector<float> x(1 * 2 * 2 * 2 * 2, 1.f), y(1 * 2 * 2 * 2 * 2, 0.f);
      st = c.forward(sess, x.data(), y.data());
      break;
    }
    case LayerId::Lstm: {
      auto l = Lstm::blank(4, 4, dt);
      std::vector<float> x(2 * 1 * 4, 0.1f), y(2 * 1 * 4, 0.f);
      st = l.forward(sess, 2, 1, x.data(), y.data());
      break;
    }
    case LayerId::Embedding: {
      auto e = Embedding::blank(8, 4, dt);
      int32_t ids[2] = {0, 1};
      std::vector<float> y(2 * 4, 0.f);
      st = e.forward(sess, 2, ids, y.data());
      break;
    }
    case LayerId::Residual: {
      auto r = Residual::blank(dt);
      std::vector<float> a = {1, 2, 3, 4}, b = {0.5f, 0.5f, 0.5f, 0.5f}, y(4, 0);
      st = r.forward(sess, 4, a.data(), b.data(), y.data());
      break;
    }
    case LayerId::SwiGlu: {
      auto g = SwiGlu::blank(4, 8, dt);
      std::vector<float> x = {1, 1, 1, 1}, y(4, 0);
      st = g.forward(sess, 1, x.data(), y.data());
      break;
    }
    case LayerId::Mha: {
      auto m = Mha::blank(8, 2, dt);
      std::vector<float> x(1 * 4 * 8, 0.1f), y(1 * 4 * 8, 0.f);
      st = m.forward(sess, 1, 4, x.data(), y.data());
      break;
    }
    default:
      st = Status::Unsupported("unknown");
      break;
  }
  check(name, st);
}

int main() {
  std::printf("welvet.mlx smoke 0.02 (forward-only, MLX GPU, no fallback)\n");
  auto d = detect();
  print_detect(d);
  print_caps();

  Session sess;
  auto st = Session::open_gpu(&sess);
  if (!st) {
    std::printf("\nFATAL: GPU session open failed — %s\n", st.message.c_str());
    std::printf("welvet.mlx refuses CPU fallback. Fix Metal/MLX and retry.\n");
    return 2;
  }
  std::printf("\nsession: ready device=%s mlx=%s\n", sess.device_name().c_str(),
              sess.mlx_version().c_str());

  std::printf("\n=== layer × dtype ===\n");
  for (int i = 0; i < (int)LayerId::Count; ++i)
    for (DType dt : kAllDtypes) run_one_dtype(sess, (LayerId)i, dt);

  std::printf("\n=== sequential dense tower ===\n");
  {
    auto net = Sequential::dense_tower(8, 3, DType::Float32);
    std::vector<float> x(8, 1.f), y(8, 0.f);
    check("sequential/dense_tower", net.forward(sess, 1, x.data(), y.data(), 1));
  }

  std::printf("\n=== summary ===\n");
  std::printf("pass=%d fail=%d skip=%d\n", g_pass, g_fail, g_skip);
  return g_fail ? 1 : 0;
}
