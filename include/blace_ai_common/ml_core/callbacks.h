#pragma once
#include "library_defines.h"
#include <tuple>

namespace blace {
namespace ml_core {
class EXPORT_OR_IMPORT ProgressCallback {
public:
  ProgressCallback();

  // called everytime a node is processed, increases internal counter
  bool set_nodes_processed(int processed, int total);

  std::tuple<int, int> get_nodes_processed() {
    return {_processed_nodes, _total_nodes};
  }

  // this is periodically checked to see if cancellation was requested
  virtual bool wants_cancel() = 0;

  // this is periodically checked to see if killing was requested
  virtual bool wants_kill() = 0;

  // called everytime a node is processed, can be overridden by user
  virtual void after_node_evaluated() = 0;

protected:
  int _processed_nodes;
  int _total_nodes;
};

class EmptyCallback : public ProgressCallback {
private:
public:
  virtual bool wants_cancel() override;

  virtual void after_node_evaluated() override;

  virtual bool wants_kill() override;
};
} // namespace ml_core
} // namespace blace