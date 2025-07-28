#pragma once

#include "blace_index.h"
#include "computation_graph/public_base_op.h"
#include "computation_graph/raw_memory_fetcher.h"
#include "computation_graph/raw_memory_object.h"
#include "ml_core/types.h"
#ifdef BLACE_AI_OPENCV_INTERFACE
#include <opencv2/core/mat.hpp>
#endif
#include <variant>

#include "library_defines.h"

typedef std::vector<int64_t> SizesVec;

namespace blace {
namespace ops {

#ifdef BLACE_AI_OPENCV_INTERFACE
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

#endif

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
  Interpolate2DOp(OpP node, int height, int width,
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
  NormalizeImagenetOp(OpP input);
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
 * Class to construct a node from a list of integers.
 */
class EXPORT_OR_IMPORT FromIntListOp : public BaseOp {
public:
  /**
   * Constructs an operator which holds an integer list.
   *
   * \param val The integer list.
   */
  FromIntListOp(std::vector<int64_t> val);
};

/**
 * Class to construct a node from a bool.
 */
class EXPORT_OR_IMPORT FromBoolOp : public BaseOp {
public:
  /**
   * Constructs an operator which holds a bool value.
   *
   * \param val The bool value.
   */
  FromBoolOp(bool val);
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
   * \param model_bytes The bytes of the model to run, will be provided from the
   * included model header. \param inputs A
   * vector of input nodes passed to the model. \param inference_args The
   * inference arguments used to run the inference. \param return_index The
   * desired result index, some models return multiple tensors. \param
   * payload_folder Payload folder.
   */
  InferenceOp(std::vector<char> model_bytes, std::vector<OpP> inputs,
              ml_core::InferenceArgsCollection inference_args, int return_index,
              std::string payload_folder);

  /**
   * Default copy constructor explicitly defined for dll exporting.
   *
   * \param e The copy class.
   */
  InferenceOp(InferenceOp const &e);
  ~InferenceOp();
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
  MapToRangeOp(OpP input, ml_core::ValueRangeEnum range);
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
  NormalizeToZeroOneOP(OpP input);

  /**
   * Constructs an operator which maps the input to the zero-one range.
   *
   * \param input The input node.
   * \param min input The input node.
   * \param max input The input node.
   */
  NormalizeToZeroOneOP(OpP input, double min, double max);

  /**
   * Constructs an operator which maps the input to the zero-one range.
   *
   * \param input The input node.
   * \param min_in input The input node.
   * \param max_in input The input node.
   * \param min_out input The input node.
   * \param max_out input The input node.
   */
  NormalizeToZeroOneOP(OpP input, OpP min_in, OpP max_in, double min_out,
                       double max_out);

  /**
   * Constructs an operator which maps the input to the zero-one range.
   *
   * \param input The input node.
   * \param min_in input The input node.
   * \param max_in input The input node.
   * \param min_out input The input node.
   * \param max_out input The input node.
   */
  NormalizeToZeroOneOP(OpP input, double min_in, double max_in, double min_out,
                       double max_out);

  /**
   * Constructs an operator which maps the input to the zero-one range.
   *
   * \param input The input node.
   * \param min_in input The input node.
   * \param max_in input The input node.
   * \param min_out input The input node.
   * \param max_out input The input node.
   */
  NormalizeToZeroOneOP(OpP input, double min_in, double max_in, OpP min_out,
                       OpP max_out);
  /**
   * Default copy constructor explicitly defined for dll exporting.
   *
   * \param e The copy class.
   */
  NormalizeToZeroOneOP(NormalizeToZeroOneOP const &e);
  ~NormalizeToZeroOneOP();
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
  ToColorOp(OpP input, ml_core::ColorFormatEnum color_format);

  /**
   * Constructs an operator which converts a tensor from one color format into
   * another.
   *
   * \param input The input node.
   * \param color_format The new color format.
   * \param lab_norms Lab norms.
   */
  ToColorOp(OpP input, ml_core::ColorFormatEnum color_format,
            ml_core::LAB_NORMS lab_norms);
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
   * \param device Put on device.
   */
  PrepareForHostCopyOP(OpP input, ml_core::DataTypeEnum data_type,
                       ml_core::ColorFormatEnum color_format,
                       ml_core::OrderEnum tensor_order,
                       ml_core::ValueRangeEnum value_range,
                       ml_core::DeviceEnum device);
};

/**
 * Operator to construct an empty tensor.
 */
class EXPORT_OR_IMPORT ZerosOp : public BaseOp {

public:
  /**
   * Operator to construct an empty tensor.
   *
   * \param sizes The sizes of the tensor.
   * \param order Tensor order.
   * \param device Device to construct the tensor on.
   * \param data_type Tensor daty type.
   * \param color_format Tensor color format.
   *
   *
   */
  ZerosOp(std::vector<int64_t> sizes, ml_core::OrderEnum order,
          ml_core::DeviceEnum device, ml_core::DataTypeEnum data_type,
          ml_core::ColorFormatEnum color_format);
  /**
   * Default copy constructor explicitly defined for dll exporting.
   *
   * \param e The copy class.
   */
  ZerosOp(ZerosOp const &e);
  ~ZerosOp();
};

/**
 * Operator to construct an empty tensor.
 */
class EXPORT_OR_IMPORT OnesOp : public BaseOp {

public:
  /**
   * Operator to construct an empty tensor.
   *
   * \param sizes The sizes of the tensor.
   * \param order Tensor order.
   * \param device Device to construct the tensor on.
   * \param data_type Tensor daty type.
   * \param color_format Tensor color format.
   *
   *
   */
  OnesOp(std::vector<int64_t> sizes, ml_core::OrderEnum order,
         ml_core::DeviceEnum device, ml_core::DataTypeEnum data_type,
         ml_core::ColorFormatEnum color_format);
};

/**
 * Operator to put values into a tensor.
 */
class EXPORT_OR_IMPORT IndexPutOp : public BaseOp {

public:
  /**
   * Puts a value from a blace::ops::OpP into the input tensor at the specified
   * indices.
   *
   * \param input
   * \param indices
   * \param val
   */
  IndexPutOp(blace::ops::OpP input, ml_core::BlaceIndexVec indices,
             blace::ops::OpP val);
  /**
   * Puts a double into the input tensor at the specified
   * indices.
   *
   * \param input
   * \param indices
   * \param val
   */
  IndexPutOp(blace::ops::OpP input, ml_core::BlaceIndexVec indices, double val);
  /**
   * Puts a float into the input tensor at the specified
   * indices.
   *
   * \param input
   * \param indices
   * \param val
   */
  IndexPutOp(blace::ops::OpP input, ml_core::BlaceIndexVec indices, float val);
  /**
   * Puts an integer into the input tensor at the specified
   * indices.
   *
   * \param input
   * \param indices
   * \param val
   */
  IndexPutOp(blace::ops::OpP input, ml_core::BlaceIndexVec indices, int val);
};

/**
 * Operator to construct an empty node.
 */
class EXPORT_OR_IMPORT NoneOp : public BaseOp {

public:
  /**
   * Constructs an empty node.
   *
   */
  NoneOp();
};

} // namespace ops
} // namespace blace