#pragma once

#include <memory>

#include "computation_graph/raw_memory_metadata.h"
#include "computation_graph/raw_memory_object.h"

namespace blace {
class RawMemoryFetcher {
public:
  virtual std::shared_ptr<RawMemoryObject> get_raw_memory_object() = 0;
  virtual std::shared_ptr<RawMemoryMetadata> get_raw_memory_metadata() = 0;
  virtual std::string printableString() = 0;
};

} // namespace blace
