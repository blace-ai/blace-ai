#pragma once

#include "ml_core/types.h"
#include <opencv2/core/mat.hpp>
#include <variant>

#include "library_defines.h"

/**
 * Use this macro to get the constructed node to be used as input for other
 * nodes.
 */
#define CONSTRUCT_OP_GET(op) op.getImpl().value()

/**
 * @file public_ops.h
 * @brief This file contains all operators that can be used to build a
 * computation graph.
 */

namespace blace {
class NodeOp;
}

namespace blace {
namespace ops {
/**
 * Base class for all operators. Don't use directly.
 */
class BaseOp {

public:
  /**
   * Returns the constructed object which can be used as input for later
   * operators.
   *
   * \return std::nullopt if construction failed, the constructed elemtent
   * otherwise.
   */
  std::optional<std::shared_ptr<NodeOp>> getImpl() { return _impl; }

  /**
   * Internal storage of the constructed object.
   */
  std::optional<std::shared_ptr<NodeOp>> _impl;
};

/**
 * Class to construct a node from a cv::Mat.
 */
class EXPORT_OR_IMPORT FromCVMatOp : public BaseOp {
public:
  /**
   * Constructs an operator from a cv::Mat.
   *
   * \param cv_mat The input cv::Mat.
   */
  FromCVMatOp(cv::Mat cv_mat);
};

/**
 * Class to construct a node from an image file on disk.
 */
class EXPORT_OR_IMPORT FromImageFileOp : public BaseOp {
public:
  /**
   * Constructs an operator from an image file on disk.
   *
   * \param path Path to the image file.
   */
  FromImageFileOp(std::string path);
};

/**
 * Interpolates a 2d tensor.
 */
class EXPORT_OR_IMPORT Interpolate2DOp : public BaseOp {
public:
  /**
   * Construct a 2d Interpolation node.
   *
   * \param node The input node.
   * \param height New height.
   * \param width New width.
   * \param interpolation Interpolation type.
   * \param align_corners Align corners, see pytorch reference implementation.
   * \param antialias Use antialiasing, see pytorch reference implementation.
   */
  Interpolate2DOp(std::shared_ptr<NodeOp> node, int height, int width,
                  ml_core::Interpolation interpolation, bool align_corners,
                  bool antialias);
};

/**
 * Normalizes the input according to the ImageNet constants.
 */
class EXPORT_OR_IMPORT NormalizeImagenetOp : public BaseOp {
public:
  /**
   * Construct a node which normalizes the input to ImageNet value range.
   *
   * \param input The input node to normalize.
   */
  NormalizeImagenetOp(std::shared_ptr<NodeOp> input);
};

/**
 * Class to construct a node from a string.
 */
class EXPORT_OR_IMPORT FromTextOp : public BaseOp {
public:
  /**
   * Constructs an operator which holds a string.
   *
   * \param text The integer value.
   */
  FromTextOp(std::string text);
};

/**
 * Class to construct a node from an integer.
 */
class EXPORT_OR_IMPORT FromIntOp : public BaseOp {
public:
  /**
   * Constructs an operator which holds an integer value.
   *
   * \param val The integer value.
   */
  FromIntOp(int val);
};

/**
 * Class to construct a node from a float.
 */
class EXPORT_OR_IMPORT FromFloatOp : public BaseOp {
public:
  /**
   * Constructs an operator which hold a float value.
   *
   * \param val The float value.
   */
  FromFloatOp(float val);
};

/**
 * Operator to run model inference.
 */
class EXPORT_OR_IMPORT InferenceOp : public BaseOp {

public:
  /**
   * Constructs an operator which runs model inference.
   *
   * \param model_ident The ident of the model to run, will be provided from the
   * included model header. \param inputs A vector of input nodes passed to the
   * model. \param inference_args The inference arguments used to run the
   * inference. \param return_index The desired result index, some models return
   * multiple tensors.
   */
  InferenceOp(ml_core::ModelIdent model_ident,
              std::vector<std::shared_ptr<NodeOp>> inputs,
              ml_core::InferenceArgsCollection inference_args,
              int return_index);
};

/**
 * Maps the values in the input operator to 0-1 range.
 */
class EXPORT_OR_IMPORT MapToRangeOp : public BaseOp {

public:
  /**
   * Constructs an operator which maps the input to 0-1 range.
   *
   * \param input The input node.
   */
  MapToRangeOp(std::shared_ptr<NodeOp> input);
};

} // namespace ops
} // namespace blace