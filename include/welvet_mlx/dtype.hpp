#pragma once

#include <cstddef>
#include <cstdint>

namespace welvet_mlx {

// Host API is float32; `dtype` selects MLX compute dtype on GPU.
// Inventory = MLX ordinary dtypes that overlap Welvet FormatNone (GPU-usable).
// Float64 exists in MLX but is CPU-only → caps reject it (no silent CPU).
enum class DType : uint8_t {
  Float32 = 0,
  Float16,
  BFloat16,
  Float64, // MLX has it; GPU unsupported
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Complex64,
};

inline const char* dtype_name(DType d) {
  switch (d) {
    case DType::Float32:
      return "float32";
    case DType::Float16:
      return "float16";
    case DType::BFloat16:
      return "bfloat16";
    case DType::Float64:
      return "float64";
    case DType::Int8:
      return "int8";
    case DType::Int16:
      return "int16";
    case DType::Int32:
      return "int32";
    case DType::Int64:
      return "int64";
    case DType::UInt8:
      return "uint8";
    case DType::UInt16:
      return "uint16";
    case DType::UInt32:
      return "uint32";
    case DType::UInt64:
      return "uint64";
    case DType::Complex64:
      return "complex64";
    default:
      return "?";
  }
}

inline size_t dtype_nbytes(DType d) {
  switch (d) {
    case DType::Float64:
    case DType::Int64:
    case DType::UInt64:
    case DType::Complex64:
      return 8;
    case DType::Float32:
    case DType::Int32:
    case DType::UInt32:
      return 4;
    case DType::Float16:
    case DType::BFloat16:
    case DType::Int16:
    case DType::UInt16:
      return 2;
    case DType::Int8:
    case DType::UInt8:
      return 1;
    default:
      return 0;
  }
}

inline bool dtype_is_float(DType d) {
  return d == DType::Float32 || d == DType::Float16 || d == DType::BFloat16 ||
         d == DType::Float64;
}

inline bool dtype_is_int(DType d) {
  switch (d) {
    case DType::Int8:
    case DType::Int16:
    case DType::Int32:
    case DType::Int64:
    case DType::UInt8:
    case DType::UInt16:
    case DType::UInt32:
    case DType::UInt64:
      return true;
    default:
      return false;
  }
}

// GPU smoke / caps sweep — Float64 included so rejection is visible.
inline constexpr DType kAllDtypes[] = {
    DType::Float32, DType::Float16, DType::BFloat16, DType::Float64,
    DType::Int8,    DType::Int16,   DType::Int32,    DType::Int64,
    DType::UInt8,   DType::UInt16,  DType::UInt32,   DType::UInt64,
    DType::Complex64,
};

enum class LayerId : uint8_t {
  Dense = 0,
  Softmax,
  RmsNorm,
  LayerNorm,
  Cnn2,
  Cnn3,
  ConvT2,
  ConvT3,
  Lstm,
  Embedding,
  Residual,
  SwiGlu,
  Mha,
  Count
};

inline const char* layer_id_name(LayerId id) {
  switch (id) {
    case LayerId::Dense:
      return "dense";
    case LayerId::Softmax:
      return "softmax";
    case LayerId::RmsNorm:
      return "rmsnorm";
    case LayerId::LayerNorm:
      return "layernorm";
    case LayerId::Cnn2:
      return "cnn2";
    case LayerId::Cnn3:
      return "cnn3";
    case LayerId::ConvT2:
      return "convt2";
    case LayerId::ConvT3:
      return "convt3";
    case LayerId::Lstm:
      return "lstm";
    case LayerId::Embedding:
      return "embedding";
    case LayerId::Residual:
      return "residual";
    case LayerId::SwiGlu:
      return "swiglu";
    case LayerId::Mha:
      return "mha";
    default:
      return "?";
  }
}

// GPU/MLX support for (layer, dtype). No silent CPU path.
bool mlx_supports(LayerId layer, DType dtype);

// True if this library has a blank+forward for the layer (any dtype).
bool layer_implemented(LayerId layer);

} // namespace welvet_mlx
