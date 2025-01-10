#pragma once

#include "ATen/core/TensorBody.h"
#include "ml_core/types.h"
#include <algorithm> // for copy
#include <cstdint>   // for int64_t
#include <vector>    // for vector

namespace blace {

typedef std::vector<int64_t> SizesVec;

/*
Stores information about some data stored in memory. This should be filled
without accessing the actual memory itself.
*/
class RawMemoryObject {
public:
  RawMemoryObject();

  RawMemoryObject(at::Tensor tensor, ml_core::ColorFormatEnum color_format,
                  ml_core::OrderEnum order);

  RawMemoryObject(void *data_ptr, ml_core::DataTypeEnum type,
                  ml_core::ColorFormatEnum color_format,
                  std::vector<int64_t> memory_sizes, ml_core::OrderEnum order,
                  bool needs_deletion);

  void *get_memory();

  std::vector<int64_t> get_memory_dimensions();
  int get_memory_size();

  ml_core::DataTypeEnum get_type();

  ml_core::ColorFormatEnum get_color_format();

  ml_core::OrderEnum get_order();

  bool get_needs_deletion();

private:
  void *_data_ptr;

  ml_core::OrderEnum _order;
  ml_core::DataTypeEnum _type;
  ml_core::ColorFormatEnum _color_format;
  std::vector<int64_t> _memory_dimensions;
  bool _needs_deletion;
};

/*
Stores information about some image stored in memory. This should be filled
without accessing the actual memory itself. It expands RawMemoryObject by a
multiplication factor.
*/
class RawImageMemoryObject : public RawMemoryObject {
public:
  RawImageMemoryObject();

  RawImageMemoryObject(void *data_ptr, ml_core::DataTypeEnum type,
                       ml_core::ColorFormatEnum color_format, double mul_fac,
                       std::vector<int64_t> memory_sizes,
                       ml_core::OrderEnum order, bool needs_deletion)
      : RawMemoryObject(data_ptr, type, color_format, memory_sizes, order,
                        needs_deletion) {
    _mul_fac = mul_fac;
  };

  double get_mul_fac();

private:
  double _mul_fac;
};

} // namespace blace