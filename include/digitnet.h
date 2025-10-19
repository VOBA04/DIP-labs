#pragma once

#include <torch/torch.h>

struct DigitNet : torch::nn::Module {
  // Слои
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr};

  DigitNet() {
    conv1 = register_module(
        "conv1", torch::nn::Conv2d(1, 16, 3)); // [1,28,28] -> [16,26,26]
    conv2 = register_module(
        "conv2", torch::nn::Conv2d(16, 32, 3)); // [16,13,13] -> [32,11,11]
    fc1 = register_module("fc1", torch::nn::Linear(32 * 5 * 5, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 10));
  }

  auto Forward(torch::Tensor x) -> torch::Tensor {
    x = torch::relu(conv1->forward(x));
    x = torch::max_pool2d(x, 2);
    x = torch::relu(conv2->forward(x));
    x = torch::max_pool2d(x, 2);
    x = x.reshape({x.size(0), -1});
    x = torch::relu(fc1->forward(x));
    x = fc2->forward(x);
    return torch::log_softmax(x, 1);
  }
};