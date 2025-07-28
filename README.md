# blace.ai inference library & c++ model hub  

Welcome to blace.ai — a high-performance C++ meta-inference library that abstracts away OS, backend (e.g., PyTorch, ONNX), and hardware accelerator differences. With a unified API and minimal setup, you can run AI models seamlessly across platforms. Explore our growing model hub for plug-and-play blace.ai-compatible models built for real-world deployment.

📦 [**download sdk**](https://github.com/blace-ai/blace-ai/releases) | 🌐 [**blace.ai website**](https://blace.ai) | 📖 [**documentation**](https://blace-ai.github.io/blace-ai/) | 🧠 [**c++ model hub**](https://www.blace.ai/hub/) | 💬 [**discord channel**](https://discord.com/channels/1202176342603616277/1318605344586338404)

⚠️ **We are currently in public beta. Therefore some features are limited and the license
only allows for experimental use. We will gradually roll out the missing features 
(mostly graph operators) and release a tool for removing the watermarks from 
models.** ⚠️

## Overview
![Screenshot](img/overview.svg) <br/><br/>
## Features
-  **Cross-Platform:** Write C++ code for ai model inference once - and deploy to 
  all major operating systems (Windows, MacOS Intel/Silicon and Linux).  
- **Backend-Agnostic** blace.ai leverages CUDA on Windows / Linux and Metal on MacOS. 
- **Self-Contained:** Our library is fully self-contained and ships with all necessary dependencies out of the box,
  making integration effortless and eliminating the hassle of managing third-party packages.
- **Performant computation graphs:** We provide you with a set of operators that can be used to write 
  computation graphs for model inference. Belows graph shows the structure of a simple graph running the Segment-Anything encoder and decoder seperately, automatically caching the intermediate results (dark orange node): <br/><br/>
  ![Screenshot](img/dag_example.svg) <br/><br/>
- **Serializable:** Built-in serialization allows you to save and transfer
  computation graphs, making it easier to integrate with distributed systems and enabling seamless scaling across multiple nodes or environments.
- **Smart models:** Models created with the [**Model Wizard**](https://blace-ai.github.io/blace-ai/md_model_wizard_creation.html) or 
  coming from the [Hub](https://www.blace.ai/hub/) store all needed configuration and 
  metadata about inputs and outputs. This eliminates the need to worry about proper model instantiation or tensor input formatting, such as memory order and sizes, streamlining the integration process.

## Quick Start  

Follow the [**Quickstart Guide**](https://blace-ai.github.io/blace-ai/md_quickstart.html#quickstart_demo) to run the first model within a few minutes.

Integrating ai models into your software should be simple. With Blace.ai, you can run AI model inference with just a few lines of code — across **Windows, Linux, and macOS**:  

```cpp
#include "blace_ai.h"

// include the models you want to use
#include "depth_anything_v2_v5_small_v3_ALL_export_version_v16.h"

cv::Mat runDepthModel() {
  workload_management::BlaceWorld blace;

  // load image into op
  auto exe_path = util::getPathToExe();
  std::filesystem::path photo_path = exe_path / "test_butterfly.jpg";
  auto world_tensor_orig =
      CONSTRUCT_OP(ops::FromImageFileOp(photo_path.string()));

  // interpolate to size consumable by model
  auto interpolated = CONSTRUCT_OP(ops::Interpolate2DOp(
      world_tensor_orig, 700, 1288, ml_core::BICUBIC, false, true));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.device = util::get_accelerator().value();

  // construct inference operation
  auto infer_op = CONSTRUCT_OP(ops::InferenceOp(
      depth_anything_v2_v5_small_v3_ALL_export_version_v16, {interpolated},
      infer_args, 0, util::getPathToExe().string()));

  // normalize depth to zero-one range
  auto result_depth = CONSTRUCT_OP(ops::NormalizeToZeroOneOP(infer_op));

  // construct evaluator and evaluate to cv::Mat
  computation_graph::GraphEvaluator evaluator(result_depth);
  auto cv_result = evaluator.evaluateToCVMat().value();

  // multiply for plotting
  cv_result *= 255.;

  // save to disk and return
  auto out_file = exe_path / "depth_result.png";
  cv::imwrite(out_file.string(), cv_result);

  return cv_result.clone();
}
```

## Installation  

Integrate blace.ai into your CMake project with just two lines.

```
include("../cmake/FindBlace.cmake")
target_link_libraries(<your_target> PRIVATE 3rdparty::BlaceAI)
```

## Roadmap 
- [**IPC Version ☂️**](https://blace-ai.github.io/blace-ai/md_ipc.html)
- add more operators
- for feature request please contact us via mail or open an issue here 

## Origin
Originally developed as the internal framework for [Blace Plugins'](https://blaceplugins.com/) AI-driven video editing tools, blace.ai has proven its reliability in production environments. Recognizing its potential to benefit a broader audience, we are thrilled to release blace.ai to the public.

## Feedback
Please open a ticket here on github. For further inquiries reach out to contact [at] blace.ai  
