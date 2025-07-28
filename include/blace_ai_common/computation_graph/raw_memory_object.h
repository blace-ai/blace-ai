#pragma once

#include "computation_graph/raw_memory_metadata.h"
#include "library_defines.h" // for EXPORT_OR_IMPORT
#include "ml_core/types.h"
#include <cstdint> // for int64_t
#include <vector>  // for vector
/**
 * @file raw_memory_object.h
 * @brief Contains implementation for a class used to I/O custom memory.
 */

namespace blace {

/**
 * Object to handle graph i/o of custom memory. The class either holds a pointer
 * to externally managed memory (not taking ownership) or it stores a copy of
 * the given memory internally.
 */

class EXPORT_OR_IMPORT RawMemoryObject {
public:
  RawMemoryObject();

  ~RawMemoryObject();

  /**
   * Constructs a RawMemoryObject from a memory pointer and meta data.
   *
   * \param data_ptr The pointer to memory on the cpu.
   * \param type Datatype of the memory.
   * \param color_format Color format of the memory.
   * \param memory_sizes A vector of the memory sizes. Total number of bytes is
   * calculated from this.
   * \param order The memory order.
   * \param value_range The value range of the passed in data. See the enum for
   * options. \param hash Hash of the data stored in memory. Make sure that
   * different data always has different hash values, otherwise caching will
   * return old values. If std::nullopt is passed in, the hash will be computed
   * based on the raw memory (safest, but also slowest option). \param
   * copy_memory If true, stores a copy of the passed memory. Otherwise the
   * constructed object only keeps a reference and the original data_ptr memory
   * has to be valid when accessed.
   */
  RawMemoryObject(void *data_ptr, ml_core::DataTypeEnum type,
                  ml_core::ColorFormatEnum color_format,
                  std::vector<int64_t> memory_sizes, ml_core::OrderEnum order,
                  ml_core::ValueRangeEnum value_range,
                  std::optional<blace::ml_core::BlaceHash> hash = std::nullopt,
                  bool copy_memory = true);

  /**
   * Constructs a RawMemoryObject from a memory pointer and meta data.
   *
   * \param data_ptr xxx
   * \param meta_data yyy
   * \param copy_memory zzz
   */
  RawMemoryObject(void *data_ptr, blace::RawMemoryMetadata meta_data,
                  bool copy_memory);

  /**
   * Copy constructor.
   *
   * \param other The other object.
   */
  RawMemoryObject(const RawMemoryObject &other);

  /**
   * Copy assignment.
   *
   * \param other The other object.
   * \return
   */
  RawMemoryObject &operator=(const RawMemoryObject &other);

  /**
   * Move constructor.
   *
   * \param other The other object.
   */
  RawMemoryObject(RawMemoryObject &&other);

  /**
   * Move assignment.
   *
   * \param other The other object.
   * \return
   */
  RawMemoryObject &operator=(RawMemoryObject &&other);

  /**
   * Get the stored memory address.
   *
   * \return
   */
  void *get_data_ptr() const;

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
  ml_core::DataTypeEnum get_type();

  /**
   * Get the color format.
   *
   * \return
   */
  ml_core::ColorFormatEnum get_color_format();

  /**
   * Get the value range.
   *
   * \return
   */
  ml_core::ValueRangeEnum get_value_range();

  /**
   * Get the memory order.
   *
   * \return
   */
  ml_core::OrderEnum get_order();

  /**
   * Get the memory hash.
   *
   * \return
   */
  ml_core::BlaceHash get_hash() const;

  /**
   * Returns if the object has memory ownership or not.
   *
   * \return
   */
  bool has_ownership();

  /**
   * Get meta data object.
   *
   * \return
   */
  RawMemoryMetadata get_meta_data() const;

  /**
   * Serializes the object for boost serialization.
   *
   */
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {}

private:
  void *_data_ptr;

  RawMemoryMetadata _memory_metadata;

  bool _take_memory_ownership = false;
};

} // namespace blace