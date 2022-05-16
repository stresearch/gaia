# GCM Integration

Refer to our deployment repo https://github.com/stresearch/gaia-deploy

## Strategy
[![](GCM_integration.png)](GCM_integration.png)

## Application Binary Interface (ABI) Development
- Start with total T,Q physics tendencies and track them back
- Check all the tendency updates from all processes
- Verify all inputs and outputs that are needed with all the tendency updates
- Find out where we want to intercept the variables
- Export Pytorch Model with Torchscript
- Bypass Python by using C++
- Call C++ within Fortran

## Exporting Pytorch Model

We export pytorch surrogate model to a torchscript format. This enables us to load a checkpoint with C++.

```C++

#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

    // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({20, 164}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

}

```
