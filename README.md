# 🚀 blace.ai inference library  

Welcome to **blace.ai inference library** — the new kid on the block for C++ AI model 
inference across multiple operating systems and hardware accelerators.  

📦 [**download sdk**](https://github.com/blace-ai/blace-ai/releases) | 🌐 [**blace.ai website**](https://blace.ai) | 📖 [**documentation**](https://blace-ai.github.io/blace-ai/) | 🧠 [**c++ model hub**](https://www.blace.ai/hub/) | 💬 [**discord channel**](https://discord.com/channels/1202176342603616277/1318605344586338404)

⚠️ **We are currently in public beta. Therefore some features are limited and the license
only allows for experimental use. We will gradually roll out the missing features 
(mostly graph operators) and release a tool for removing the watermarks from 
models.** ⚠️

## 🎯 Features
- 🖥️ **Cross-Platform:** Write C++ code for ai model inference once - and deploy to 
  all major operating systems (Windows, MacOS Intel/Silicon and Linux).  
- 🔌 **Extensible:** We provide you with a set of operators that can be used to write 
  cacheable and serializable computation graphs for model inference. The framework 
  takes care of model loading, memory management, graph optimization, and caching of intermediate results.
- 📦 **Smart models:** Models created with our converter (to be released soon) or 
  coming from the [Hub](https://www.blace.ai/hub/) store all needed configuration and 
  metadata about inputs and outputs. This eliminates the need to worry about proper model instantiation or tensor input formatting, such as memory order and sizes, streamlining the integration process.

## 🚀 Quick Start  

Follow the [**Quickstart Guide**](https://blace-ai.github.io/blace-ai/md_quickstart.html#quickstart_demo) to run the first model within a few minutes.

Integrating ai models into your software should be simple. With blace.ai inferencing e.g., googles gemma model can be achieved
with a few lines of code. Again, this will work across **Windows, Linux and MacOS**:  

```cpp
// model header stores the metadata payload and identifier needed to run it
#include "gemma_v1_default_v1_ALL_export_version_v10.h"
// register the model
blace::util::registerModel(gemma_v1_default_v1_ALL_export_version_v10, blace::util::getPathToExe());

// construct model inputs
auto text_t = CONSTRUCT_OP_GET(blace::ops::FromTextOp("Which is your favorite lord of 
the rings movie?"));
auto output_len = CONSTRUCT_OP_GET(blace::ops::FromIntOp(200));
auto temperature = CONSTRUCT_OP_GET(blace::ops::FromFloatOp(0.));
auto top_p = CONSTRUCT_OP_GET(blace::ops::FromFloatOp(0.9));
auto top_k = CONSTRUCT_OP_GET(blace::ops::FromIntOp(50));

// setup inference arguments
blace::ml_core::InferenceArgsCollection infer_args;
// get available accelerator (cuda or metal device, if available)
infer_args.inference_args.device = blace::util::get_accelerator().value();
 
// construct inference operator
auto infer_op = CONSTRUCT_OP_GET(blace::ops::InferenceOp(gemma_v1_default_v1_ALL_export_version_v10_IDENT,
            {text_t, output_len, temperature, top_p, top_k}, infer_args, 0));

// evaluate and print the result
blace::computation_graph::GraphEvaluator evaluator(infer_op);
auto answer = evaluator.evaluateToString().value();
std::cout << "Answer: " << answer << std::endl;
}
```

## 📥 Installation  

Integrate blace.ai into your CMake project with just two lines.

```
include("../cmake/FindBlace.cmake")
target_link_libraries(<your_target> PRIVATE 3rdparty::BlaceAI)
```

## 🧭 Roadmap:
- add missing operators
- provide a tool to convert existing torchscript and onnx models into our framework
- provide a tool to remove watermark from models coming from our hub

## 🌱 Origin:
Originally developed as the internal framework for [Blace Plugins'](https://blaceplugins.com/) AI-driven video editing tools, blace.ai has proven its reliability in production environments. Recognizing its potential to benefit a broader audience, we are thrilled to release blace.ai to the public.

## 📭 Feedback
Please open a ticket here on github. For further inquiries reach out to contact [at] blace.ai  
