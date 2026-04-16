#pragma once

#include <memory>

#include "computation_graph/raw_memory_metadata.h"
#include "computation_graph/raw_memory_object.h"
#include <iostream>

/**
 * @file raw_memory_fetcher.h
 * @brief Contains implementation for a class used to I/O custom memory lazily.
 */

namespace blace {
/**
 * Interface class to create on object which will provide a RawMemoryObject upon
 * request.
 */
class RawMemoryFetcher {
public:
  /**
   * Returns a pair of error code and shared pointer of a RawMemoryObject. If
   * the returned code is not blace::ml_core::ReturnCode::OK, the pointer can be
   * null. If return code is blace::ml_core::ReturnCode::OK the returned pointer
   * has to hold a valid object. Might be called during graph evaluation, but
   * caching might prevent a call to this at all.
   *
   * \return
   */
  virtual std::pair<ml_core::ReturnCode, std::shared_ptr<RawMemoryObject>>
  get_raw_memory_object() = 0;
  /**
   * Return the meta data associated with the RawMemoryObject. Will be called
   * during graph construction.
   *
   * \return
   */
  virtual std::shared_ptr<RawMemoryMetadata> get_raw_memory_metadata() = 0;
  /**
   * Return an additional string used in debug logging.
   *
   * \return
   */
  virtual std::string printableString() = 0;
};

} // namespace blace
