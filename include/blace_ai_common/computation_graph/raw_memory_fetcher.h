#pragma once

#include <memory>

#include "computation_graph/raw_memory_metadata.h"
#include "computation_graph/raw_memory_object.h"
#include <iostream>
namespace blace {
class RawMemoryFetcher {
public:
  virtual std::pair<ml_core::ReturnCode, std::shared_ptr<RawMemoryObject>>
  get_raw_memory_object() = 0;
  virtual std::shared_ptr<RawMemoryMetadata> get_raw_memory_metadata() = 0;
  virtual std::string printableString() = 0;
  virtual bool is_memory_ready() { return true; }
  virtual uint32_t memory_identifier() { return 0; }
};

class DeserializedRawMemoryFetcher : public RawMemoryFetcher {
public:
  DeserializedRawMemoryFetcher(RawMemoryMetadata metadata);
  virtual std::pair<ml_core::ReturnCode, std::shared_ptr<RawMemoryObject>>
  get_raw_memory_object() override;
  virtual std::shared_ptr<RawMemoryMetadata> get_raw_memory_metadata() override;
  virtual std::string printableString() override;

private:
  RawMemoryMetadata metadata;
};

} // namespace blace
