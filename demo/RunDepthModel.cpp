#include "RunDepthModel.h"
#include "blace_ai.h"

// include the models you want to use
#include "depth_anything_v2_v5_small_v3_ALL_export_version_v16.h"

cv::Mat blace::testing::runDepthModel() {

  // register model at server
  blace::util::registerModel(
      depth_anything_v2_v5_small_v3_ALL_export_version_v16,
      blace::util::getPathToExe());

  // load image into op
  auto exe_path = blace::util::getPathToExe();
  std::filesystem::path photo_path = exe_path / "test_butterfly.jpg";
  auto world_tensor_orig =
      CONSTRUCT_OP(blace::ops::FromImageFileOp(photo_path.string()));

  // interpolate to size consumable by model
  auto interpolated = CONSTRUCT_OP(
      blace::ops::Interpolate2DOp(world_tensor_orig, 700, 1288,
                                  ml_core::BICUBIC, false, true));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.device = blace::util::get_accelerator().value();

  // construct inference operation
  auto infer_op = CONSTRUCT_OP(blace::ops::InferenceOp(
      depth_anything_v2_v5_small_v3_ALL_export_version_v16_IDENT,
      {interpolated}, infer_args, 0));

  // normalize depth to zero-one range
  auto result_depth =
      CONSTRUCT_OP(blace::ops::NormalizeToZeroOneOP(infer_op));

  // construct evaluator and evaluate to cv::Mat
  computation_graph::GraphEvaluator evaluator(result_depth);
  auto cv_result = evaluator.evaluateToCVMat().value();

  // multiply for plotting
  cv_result *= 255.;

  // save to disk and return
  auto out_file = exe_path / "depth_result.png";
  cv::imwrite(out_file.string(), cv_result);

  // unload all models before program exits
  blace::util::unloadModels();

  return cv_result.clone();
}
