#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>
#include "models/PhysicsInformedNN.hpp"

int main(){
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA GPU" : "CPU") << std::endl;

    std::cout << "Training Physics-Informed Neural Network..." << std::endl;

    PinnTrainer trainer(S, K, T, r, sigma, opt_type);
    // trainer.train(1000, 1e-3);

}