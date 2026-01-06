#include "RunDepthModel.h"
#include "blace_ai.h"
#include <opencv2/opencv.hpp>

// include the models you want to use
#include "depth_anything_v2_v8_small_v3_ALL_export_version_v17.h"

namespace blace {

/**
 * Reads and image from disk and puts the image data into a
 * blace::RawMemoryObject which can be feed to the computation graph.
 *
 * \param file
 * \return
 */
std::shared_ptr<RawMemoryObject> raw_memory_from_file(std::string file) {
  // read image into memory
  cv::Mat image = cv::imread(file, cv::IMREAD_COLOR);

  // construct a random hash from the filename
  std::hash<std::string> seeder;
  int seed = seeder(file);
  blace::ml_core::BlaceHash random_hash(seed);

  blace::RawMemoryObject raw_mem(
      (void *)image.data, blace::ml_core::DataTypeEnum::BLACE_BYTE,
      blace::ml_core::ColorFormatEnum::BGR,
      std::vector<int64_t>{1, image.rows, image.cols, 3}, blace::ml_core::BHWC,
      blace::ml_core::ZERO_TO_255, ml_core::CPU, random_hash, true);

  return std::make_shared<blace::RawMemoryObject>(raw_mem);
}

/**
 * Compares two cv::Mats for equality.
 *
 * \param a
 * \param b
 */
void compareToRef(std::string a, std::string b) {
  cv::Mat a_mat = cv::imread(a, cv::IMREAD_GRAYSCALE);

  // load reference value and compare
  auto b_mat = cv::imread(b, cv::IMREAD_GRAYSCALE);
  b_mat.convertTo(b_mat, a_mat.type());

  cv::Mat diff;
  cv::absdiff(a_mat, b_mat, diff);

  cv::Scalar meanSquareError = cv::mean(diff);

  if (meanSquareError[0] > 1.) {
    std::cerr << "Values don't match" << std::endl;
    throw("Values don't match");
  }
}

void runDepthModel(std::string input_file, std::string output_file) {
  // have this present globally
  workload_management::BlaceWorld blace;

  // load image into op
  auto image_mem = raw_memory_from_file(input_file);
  auto world_tensor_orig = CONSTRUCT_OP(blace::ops::FromRawMemoryOp(image_mem));

  // interpolate to smaller size
  auto interpolated = CONSTRUCT_OP(ops::Interpolate2DOp(
      world_tensor_orig, 700, 1288, ml_core::BICUBIC, false, true));

  // construct model inference arguments for cross-platform compatibility
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::ONNX_DML_FP32, ml_core::TORCHSCRIPT_CPU_FP32};

  // construct inference operation
  auto infer_op = CONSTRUCT_OP(ops::InferenceOp(
      depth_anything_v2_v8_small_v3_ALL_export_version_v17, {interpolated},
      infer_args, 0, util::getPathToExe().string()));

  // normalize depth to zero-one range
  auto result_depth = CONSTRUCT_OP(ops::NormalizeToZeroOneOP(infer_op));

  // save result to image file
  result_depth = CONSTRUCT_OP(ops::SaveImageOp(result_depth, output_file));

  // construct (IPC) evaluator and evaluate to raw memory object
#ifndef BLACE_AI_USE_IPC
  computation_graph::GraphEvaluator evaluator(result_depth);
  auto eval_result = evaluator.evaluateToRawMemory();
#else
  ipc::IpcEvaluator ipc_evaluator(result_depth);
  auto eval_result = ipc_evaluator.evaluateToRawMemory();
#endif

  assert(eval_result.first == ml_core::ReturnCode::OK);

  return;
}

void run() {
  auto exe_path = util::getPathToExe();
  std::filesystem::path input_path = exe_path / "test_butterfly.jpg";
  std::filesystem::path output_path = exe_path / "depth_result.png";
  std::filesystem::path ref_path = exe_path / "depth_result_ref.png";

  runDepthModel(input_path.string(), output_path.string());

  compareToRef(output_path.string(), ref_path.string());
}

} // namespace blace