#pragma once
#include <vector>

#include "ml_core/types.h" // for basic_types::ColorFormatMessage

namespace blace {

class EXPORT_OR_IMPORT RawMemoryMetadata {
public:
  RawMemoryMetadata();

  RawMemoryMetadata(ml_core::OrderEnum order, ml_core::DataTypeEnum type,
                    ml_core::ColorFormatEnum color_format,
                    ml_core::ValueRangeEnum norm,
                    std::vector<int64_t> memory_sizes,
                    blace::ml_core::BlaceHash hash,
                    blace::ml_core::DeviceEnum device);

  /**
   * Get the vector of memory sizes. E.g. a blace::ml_core::BCHW
   * blace::ml_core::RGB tensor with sizes 512x512 will return {1,3,512,512}.
   *
   * \return
   */
  std::vector<int64_t> get_memory_sizes() const;

  /**
   * Get the memory size in bytes. Computed from the memory sizes and data type.
   *
   * \return
   */
  int get_memory_size() const;

  /**
   * Get the data type.
   *
   * \return
   */
  ml_core::DataTypeEnum get_type() const;

  /**
   * Get the color format.
   *
   * \return
   */
  ml_core::ColorFormatEnum get_color_format() const;

  /**
   * Get the value range.
   *
   * \return
   */
  ml_core::ValueRangeEnum get_value_range() const;

  /**
   * Get the memory order.
   *
   * \return
   */
  ml_core::OrderEnum get_order() const;

  /**
   * Get the memory hash.
   *
   * \return
   */
  ml_core::BlaceHash get_hash() const;

  ml_core::DeviceEnum get_device() const;

  void set_hash(ml_core::BlaceHash hash);

  void set_memory_sizes(std::vector<int64_t> sizes);

  static int calc_memory_size(std::vector<int64_t> sizes,
                              ml_core::DataTypeEnum type);

private:
  ml_core::OrderEnum _order;
  ml_core::DataTypeEnum _type;
  ml_core::ColorFormatEnum _color_format;
  ml_core::ValueRangeEnum _norm;
  std::vector<int64_t> _memory_sizes;
  ml_core::BlaceHash _hash;
  ml_core::DeviceEnum _device;
};

} // namespace blace