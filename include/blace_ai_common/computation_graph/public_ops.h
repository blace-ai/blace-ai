#pragma once

#include "computation_graph/raw_memory_object.h"
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
 * Maps the values in the input operator to a given range based on the meta data
 * of the input operator. If the input operator has range
 * ml_core::ValueRangeEnum::UNKNOWN_VALUE_RANGE the operator will fail.
 */
class EXPORT_OR_IMPORT MapToRangeOp : public BaseOp {

public:
  /**
   * Constructs an operator which maps the input to the given range.
   *
   * \param input The input node.
   * \param range The requested range.
   */
  MapToRangeOp(std::shared_ptr<NodeOp> input, ml_core::ValueRangeEnum range);
};

/**
 * Maps the values in the input operator to a the zero-one range based on the
 * minimum and maximum values of the input operator.
 */
class EXPORT_OR_IMPORT NormalizeToZeroOneOP : public BaseOp {

public:
  /**
   * Constructs an operator which maps the input to the zero-one range.
   *
   * \param input The input node.
   */
  NormalizeToZeroOneOP(std::shared_ptr<NodeOp> input);
};

/**
 * Converts a tensor from one color format into another.
 */
class EXPORT_OR_IMPORT ToColorOp : public BaseOp {

public:
  /**
   * Constructs an operator which converts a tensor from one color format into
   * another.
   *
   * \param input The input node.
   * \param color_format The new color format.
   */
  ToColorOp(std::shared_ptr<NodeOp> input,
            ml_core::ColorFormatEnum color_format);
};

/**
 * Operator to construct a node from custom memory.
 */
class EXPORT_OR_IMPORT FromRawMemoryOp : public BaseOp {

public:
  /**
   * Constructs an operator based on custom memory (self or externally managed).
   * The custom memory holds all metadata like sizes, datatype etc. to allow for
   * full tracking of metadata inside the graph.
   *
   * \param mem_object A shared_ptr of the memory object.
   */
  FromRawMemoryOp(std::shared_ptr<blace::RawMemoryObject> mem_object);
};

/**
 * Operator to prepare tensor for copying to host.
 */
class EXPORT_OR_IMPORT PrepareForHostCopyOP : public BaseOp {

public:
  /**
   * Prepares a the internal tensor for copying to host by making sure all meta
   * data (type, ordering, sizes etc.) are aligned.
   *
   * This needs to be used together with GraphEvaluator::evaluateToRawMemory()
   * often. E.g. a cv::Mat will expect BGR color in a HWC tensor.
   *
   * \param input
   * \param data_type The desired datatype.
   * \param color_format Target color format.
   * \param tensor_order Target tensor order.
   * \param value_range Target value range.
   */
  PrepareForHostCopyOP(std::shared_ptr<NodeOp> input,
                       ml_core::DataTypeEnum data_type,
                       ml_core::ColorFormatEnum color_format,
                       ml_core::OrderEnum tensor_order,
                       ml_core::ValueRangeEnum value_range);
};

} // namespace ops
} // namespace blace