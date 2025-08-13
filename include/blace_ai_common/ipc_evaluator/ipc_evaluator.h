#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>

#include "computation_graph/public_base_op.h"
#include "computation_graph/raw_memory_object.h"
#include <filesystem>
#include <functional>
#include <list>
#include <thread>

/**
 * @file ipc_evaluator.h
 * @brief Contains methods to evaluate the constructed graph in a seperate
 * process.
 */

namespace blace {
namespace ipc {

class IpcEvaluatorImpl;

/**
 * A class to evaluate constructed graphs in an IPC process.
 */
class IpcEvaluator {
public:
  IpcEvaluator();

  /**
   * Setup the IPC worker. Needs to be called before graph can be processed.
   *
   */
  void setup();

  /**
   * Destruct the IPC process.
   *
   */
  void destruct();

  /**
   * Evalue a computation graph to raw memory in an seperate process.
   *
   * \param computation_graph
   * \return
   */
  std::shared_ptr<::blace::RawMemoryObject>
  evaluateToRawMemory(::blace::ops::OpP computation_graph);

private:
  std::shared_ptr<IpcEvaluatorImpl> impl;
};

} // namespace ipc
} // namespace blace