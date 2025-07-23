#pragma once

#include "library_defines.h"
#include <cstdint>
#include <iosfwd> // for ostream
#include <optional>
#include <string>
#include <vector>

/**
 * @file types.h
 * @brief Basic types of the blace.ai library.
 */

namespace blace {
namespace ml_core {

/**
 * Tensor sizes.
 */
typedef std::vector<int64_t> TensorSizes;

/**
 * Maximum value of a byte.
 */
constexpr int BYTE_MAX = 255;
/**
 * Maximum value of a short.
 */
constexpr float SHORT_MAX = 32768.;
/**
 * Maximum value of a byte as float.
 */
constexpr float BYTE_MAX_FLOAT = 255.;

/**
 * Device enumeration. Use CPU for cpu inference, CUDA for cuda accelerated
 * inference on Windows and Linux systems with NVidia GPU and MPS for metal
 * acceleration on MacOS.
 */
enum DeviceEnum : int { CPU = 0, CUDA = 1, MPS = 2 };

/**
 * Enum with different options for value ranges inside a tensor.
 */
enum ValueRangeEnum : int {
  ZERO_TO_ONE = 0,
  MINUS_ONE_TO_ONE = 1,
  IMAGENET = 2,
  UNKNOWN_VALUE_RANGE = 3,
  ZERO_TO_255 = 4,
  MINUS_0_5_TO_0_5 = 5,
  ZERO_TO_32768 = 6
};

/**
 * Data types of a tensor.
 */
enum DataTypeEnum : int {
  INT_32 = 0,
  FLOAT_32 = 1,
  BLACE_BYTE = 2,
  BLACE_BOOL = 3,
  FLOAT_32_16 = 4,
  FLOAT_16 = 5,
  SHORT = 6,
  INT_64 = 7,
  FLOAT_64 = 8
};

/**
 * Order of a tensor.
 */
enum OrderEnum : int {
  BTCHW = 0,
  BCHW = 1,
  CHW = 2,
  HWC = 3,
  BHWC = 4,
  HW = 5,
  W = 6,
  WC = 7,
  C = 8,
  BC = 9,
  BWCH = 10,
  BHW = 11,
  BCH = 12,
  CH = 13,
  TBCHW = 14,
  BCWH = 15,
  BWHC = 16,
  NO_DIMS = 17,
  UNKNOWN_ORDER = 18,
  BOUNDING_BOX_WITH_DIMS = 19,
  THWC = 20,
  TCHW = 21

};

/**
 * Color format of a tensor.
 */
enum ColorFormatEnum : int {
  RGB = 0,
  R = 1,
  A = 2,
  ARGB = 3,
  ARBITRARY_CHANNELS = 4,
  BGRA = 5,
  BGR = 6,
  LAB = 7,
  AB = 8,
  L = 9,
  XYZ = 10,
  YCBCR = 11,
  Y = 12,
  BRG = 13,
  UV = 14
};

/**
 * Model data type.
 */
enum ModelDataType { tFloat = 0, tHalf };

/**
 * Inputs to for LAB color space conversion.
 */
struct LAB_NORMS {
  /**
   * Undocumented.
   */
  double l_norm;
  /**
   * Undocumented.
   */
  double l_cent;
  /**
   * Undocumented.
   */
  double ab_norm;

  /**
   * Simple equality operator.
   *
   * \param other The other object.
   * \return
   */
  bool operator==(const LAB_NORMS &other) const;
};

/**
 * Model inference modes.
 */
enum FORWARD_MODE { SIMPLE = 0, MULTIPASS, SLICED };

/**
 * Tensor interpolation modes.
 */
enum Interpolation {
  NEAREST = 0,
  LINEAR,
  BILINEAR,
  BICUBIC,
  TRILINEAR,
  AREA,
  PIL_BICUBIC
};

/**
 * Tensor padding modes.
 */
enum PADDING_MODE { REPLICATION = 0, REFLECTION };

/**
 * Not used.
 */
enum DIRECTION { LEFT, TOP, RIGHT, BOTTOM };

/**
 * Struct to hold a hash.
 */
struct EXPORT_OR_IMPORT BlaceHash {
  /**
   * The internal value store.
   */
  uint64_t hash[4];

  /**
   * Equality operator.
   *
   * \param rhs The other hash.
   * \return true if equal.
   */
  bool operator==(const BlaceHash &rhs) const;

  /**
   * Inequality operator.
   *
   * \param rhs The other hash.
   * \return true if equal.
   */
  bool operator!=(const BlaceHash &rhs) const;

  /**
   * Convert hash to a long long.
   *
   * \return the long long
   */
  long long to_long_long();

  /**
   * Convert hash to a hexadecimal string.
   *
   * \param len Length of the string.
   * \return The hex string.
   */
  std::string to_hex(int len = 8);

  /**
   * Print helper.
   */
  friend std::ostream &operator<<(std::ostream &os, const BlaceHash &obj);

  /**
   * Default constructor.
   *
   */
  BlaceHash();

  /**
   * Construct hash from a seed.
   *
   * \param seed The seed
   */
  BlaceHash(int seed);

  /**
   * Construct random from a string.
   *
   * \param str The string to create the hash from.
   */
  BlaceHash(std::string str);

  /**
   * Construct from 4 values.
   *
   * \param a A
   * \param b B
   * \param c C
   * \param d D, who would have thought?
   */
  BlaceHash(int64_t a, int64_t b, int64_t c, int64_t d);

  /**
   * Simple print helper.
   *
   * \return
   */
  std::string print() {
    std::string returnstring = "";
    for (int temp = 0; temp < 4; temp++)
      returnstring += std::to_string(hash[temp]) + "\n";
    return returnstring;
  }

  /**
   * x
   *
   * \param hash x
   * \param str x
   */
  static void mix_string_into_hash(BlaceHash &hash, std::string str);

  /**
   * x
   *
   * \param hash x
   * \param data x
   */
  static void mix_float_into_hash(BlaceHash &hash, float data);
};

/**
 * Model inference arguments.
 */
struct ModelInferenceArgs {
  /**
   * The device to run the inference on.
   */
  DeviceEnum device = DeviceEnum::CPU;
  /**
   * Inference in half (fp16) precision if possible.
   */
  int use_half = true;
  /**
   * A value to seed random operators with.
   */
  int seed = 0;
  /**
   * Run inference in a seperate thread which can be cancelled (not available in
   * beta).
   */
  int run_threaded = false;
  /**
   * Not used.
   */
  int plot_inputs = false;
  /**
   * If several models are invoked in computation, unload all models to the cpu
   * and have only current model on hardware accelerator.
   */
  int gpu_mem_opti = false;
  /**
   * Empty backend (cuda or metal) caches after every inference, might save some
   * memory.
   */
  int empty_cache_after_inference = false;

  /**
   * Run torchscript model in autocast mode.
   */
  int experimental_torchscript_autocast = false;

  /**
   * Hash the struct.
   *
   * \return
   */
  ml_core::BlaceHash hash();

  /**
   * Simple equality operator.
   *
   * \param other The other struct.
   * \return
   */
  bool operator==(const ModelInferenceArgs &other) const;

  /**
   * Simple inequality operator.
   *
   * \param other The other struct.
   * \return
   */
  bool operator!=(const ModelInferenceArgs &other) const {
    return !(*this == other);
  }
};

/**
 * Model multisample inference arguments.
 */
struct MultisampleInferenceArgs {
  /**
   * Run all samples in parallel.
   */
  int parallel = true;
  /**
   * Number of samples to run.
   */
  int samples = 1;
  /**
   * Undocumented.
   */
  Interpolation jitter_interpolation = BILINEAR;
  /**
   * Undocumented.
   */
  int jitter_keep_size = false;
  /**
   * Undocumented.
   */
  int max_extension = 32;
  /**
   * Undocumented.
   */
  PADDING_MODE padding_mode = PADDING_MODE::REFLECTION;
  /**
   * Undocumented.
   */
  int result_to_input_num = 1;
  /**
   * Undocumented.
   */
  int result_to_input_denum = 1;

  /**
   * Hash the struct.
   *
   * \return
   */
  ml_core::BlaceHash hash();

  /**
   * Simple equality operator.
   *
   * \param other The other struct.
   * \return
   */
  bool operator==(const MultisampleInferenceArgs &other) const;

  /**
   * Simple inequality operator.
   *
   * \param other The other struct.
   * \return
   */
  bool operator!=(const MultisampleInferenceArgs &other) const {
    return !(*this == other);
  }
};

/**
 * Sliced inference arguments.
 */
struct SlicedInferenceArgs {
  /**
   * Run slices in parallel.
   */
  int parallel = true;
  /**
   * Undocumented.
   */
  int slices = 1;
  /**
   * Undocumented.
   */
  int overlap = 32;
  /**
   * Undocumented.
   */
  int result_to_input_num = 1;
  /**
   * Undocumented.
   */
  int result_to_input_denum = 1;

  /**
   * Hash the struct.
   *
   * \return
   */
  ml_core::BlaceHash hash();

  /**
   * Simple equality operator.
   *
   * \param other The other struct.
   * \return
   */
  bool operator==(const SlicedInferenceArgs &other) const;

  /**
   * Simple inequality operator.
   *
   * \param other The other struct.
   * \return
   */
  bool operator!=(const SlicedInferenceArgs &other) const {
    return !(*this == other);
  }
};

/**
 * A collection of arguments holding all parameters for model inference.
 */
class InferenceArgsCollection {
public:
  /**
   * The basic inference arguments like backend etc.
   */
  ModelInferenceArgs inference_args;
  /**
   * The mode the inference shall be run in. MULTIPASS will run multiple
   * jittered passes on BCHW tensors and SLICED cuts the inputs in parts to save
   * memory.
   */
  ml_core::FORWARD_MODE mode = FORWARD_MODE::SIMPLE;
  /**
   * In case of MULTIPASS hold the arguments here.
   */
  std::optional<MultisampleInferenceArgs> multi_args = std::nullopt;
  /**
   * In case of SLICED hold the arguments here.
   */
  std::optional<SlicedInferenceArgs> sliced_args = std::nullopt;

  /**
   * Hash the struct.
   *
   * \return The hash.
   */
  ml_core::BlaceHash hash();

  /**
   * Simple equality operator.
   *
   * \param other The other struct.
   * \return
   */
  bool operator==(const InferenceArgsCollection &other) const;

  /**
   * Simple inequality operator.
   *
   * \param other The other struct.
   * \return
   */
  bool operator!=(const InferenceArgsCollection &other) const {
    return !(*this == other);
  }
};

/**
 * Wrapper around string to identify a model.
 */
class ModelIdent : public std::string {
public:
  using std::string::string;

  /**
   * Construct a ModelIdent from a string.
   *
   * \param str The input string.
   */
  ModelIdent(const std::string &str) : std::string(str) {
    // Additional initialization or processing can be done here
  }
};

} // namespace ml_core
} // namespace blace