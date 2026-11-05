#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "welvet_mlx/layers.hpp"
#include "welvet_mlx/session.hpp"
#include "welvet_mlx/status.hpp"

namespace welvet_mlx {

using LayerOp = std::variant<Dense, Softmax, RmsNorm, LayerNorm, SwiGlu, Mha, Lstm>;

// Sequential: child[0]→…→child[n] on the GPU session (each child its own MLX eval).
class Sequential {
 public:
  Sequential() = default;

  Status push(Dense d);
  Status push(Softmax s);
  Status push(RmsNorm r);
  Status push(LayerNorm l);
  Status push(SwiGlu s);
  Status push(Mha m);
  Status push(Lstm lstm);

  void clear();
  size_t size() const { return ops_.size(); }
  bool empty() const { return ops_.empty(); }

  int32_t in_features() const { return in_features_; }
  int32_t out_features() const { return out_features_; }

  const std::vector<LayerOp>& ops() const { return ops_; }

  Status forward(Session& sess, int32_t batch, const float* x, float* y, int32_t seq = 1) const;

  static Sequential dense_tower(int32_t dim, int32_t depth, DType dt = DType::Float32);

 private:
  Status push_checked(LayerOp op, int32_t expect_in, int32_t produce_out);
  Status forward_one(Session& sess, const LayerOp& op, int32_t batch, int32_t seq, const float* x,
                     float* y) const;

  std::vector<LayerOp> ops_;
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;
};

} // namespace welvet_mlx
