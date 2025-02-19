#pragma once
#include "computation_graph/raw_memory_object.h"
#include "library_defines.h"
#include "ml_core/callbacks.h"
#include <filesystem>
#include <memory>
#include <opencv2/core/mat.hpp>

/**
 * @file graph_evaluator.h
 * @brief Contains methods to evaluate the constructed graph.
 */

namespace blace {
class NodeOp;
}

namespace blace {
namespace computation_graph {
/**
 * Internal struct, cannot be used.
 */
struct GraphEvaluatorImpl;

/**
 * A class to evaluate constructed graphs.
 */
class EXPORT_OR_IMPORT GraphEvaluator {
public:
  /**
   * Constructs a graph evaluator instance from a node.
   *
   * \param node The final node which shall be evaluated.
   */
  GraphEvaluator(std::shared_ptr<NodeOp> node);

  /**
   * Writes a .dot file with the graph structure.
   *
   * \param filename
   */
  void to_dot_file(const std::filesystem::path &filename);

  /**
   * Evaluates graph to a cv::Mat.
   *
   * \return cv::Mat result or std::nullopt in case of error.
   */
  std::optional<cv::Mat> evaluateToCVMat();

  /**
   * Evaluates graph to string.
   *
   * \return std::string result or std::nullopt in case of error.
   */
  std::optional<std::string> evaluateToString();

  /**
   * Evaluates graph to raw memory.
   *
   * \return RawMemoryObject result or std::nullopt in case of error.
   */
  std::optional<RawMemoryObject> evaluateToRawMemory();

  /**
   * Evaluates graph to at::Tensor.
   *
   * \return at::Tensor result or std::nullopt in case of error.
   */
  std::optional<at::Tensor> evaluateToTorchTensor(
      std::shared_ptr<ml_core::ProgressCallback> progress_callback = nullptr);

  ~GraphEvaluator();

private:
  std::unique_ptr<GraphEvaluatorImpl> _impl;
};
} // namespace computation_graph
} // namespace blace
