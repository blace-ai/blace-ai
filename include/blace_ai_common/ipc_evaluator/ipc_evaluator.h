#pragma once
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
class EXPORT_OR_IMPORT IpcEvaluator {
public:
  /**
   * Construct evaluator with a graph to evaluate.
   * \param computation_graph graph to be evaluated in worker.
   */
  IpcEvaluator(::blace::ops::OpP computation_graph);

  /**
   * Evalue a computation graph to raw memory in an seperate process.
   * \return evaluated graph.
   */
  std::optional<std::shared_ptr<RawMemoryObject>> evaluateToRawMemory();

private:
  ::blace::ops::OpP computation_graph;
  std::shared_ptr<IpcEvaluatorImpl> impl;
};

} // namespace ipc
} // namespace blace