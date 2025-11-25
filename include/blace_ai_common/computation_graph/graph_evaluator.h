#pragma once
#include "computation_graph/raw_memory_object.h"
#include "library_defines.h"
#include "ml_core/callbacks.h"
#include "public_ops.h"
#ifdef BLACE_AI_TORCH_INTERFACE
#include <ATen/core/TensorBody.h>
#endif
#include <filesystem>
#include <memory>
#ifdef BLACE_AI_OPENCV_INTERFACE
#include <opencv2/core/mat.hpp>
#endif

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
   * \param is_in_worker If it is inside an IPC worker.
   */
  GraphEvaluator(blace::ops::OpP node, bool is_in_worker = false);

  /**
   * Writes a .dot file with the graph structure.
   *
   * \param filename
   */
  void to_dot_file(const std::filesystem::path &filename);

  /**
   * Returns a .dot string with the graph structure.
   *
   */
  std::string to_dot_string();

#ifdef BLACE_AI_OPENCV_INTERFACE
  /**
   * Evaluates graph to a cv::Mat.
   *
   * \return cv::Mat result or std::nullopt in case of error.
   */
  std::pair<ml_core::ReturnCode, cv::Mat> evaluateToCVMat();
#endif

  /**
   * Evaluates graph to string.
   *
   * \return std::string result or std::nullopt in case of error.
   */
  std::pair<ml_core::ReturnCode, std::string> evaluateToString();

  /**
   * Evaluates graph to an integer list.
   *
   * \return std::vector<int64_t> result or std::nullopt in case of error.
   */
  std::pair<ml_core::ReturnCode, std::vector<int64_t>> evaluateToIntList();

  /**
   * Evaluates graph to at::Tensor.
   *
   * \return at::Tensor result or std::nullopt in case of error.
   */
  std::pair<ml_core::ReturnCode, std::shared_ptr<RawMemoryObject>>
  evaluateToRawMemory(
      std::shared_ptr<ml_core::ProgressCallback> progress_callback = nullptr);

#ifdef BLACE_AI_TORCH_INTERFACE
  /**
   * Evaluates graph to at::Tensor.
   *
   * \return at::Tensor result or std::nullopt in case of error.
   */
  bool has_all_required_data();

  /**
   * Evaluates graph to at::Tensor.
   *
   * \return at::Tensor result or std::nullopt in case of error.
   */
  std::vector<int64_t> missing_data_frames();

  // at::Tensor evaluateToTorchTensor(
  //     std::shared_ptr<ml_core::ProgressCallback> progress_callback =
  //     nullptr);
  std::pair<ml_core::ReturnCode, at::Tensor> evaluateToTorchTensor(
      std::shared_ptr<ml_core::ProgressCallback> progress_callback = nullptr);
#endif

  ~GraphEvaluator();

private:
  std::unique_ptr<GraphEvaluatorImpl> _impl;
  bool is_in_worker;
};
} // namespace computation_graph
} // namespace blace
