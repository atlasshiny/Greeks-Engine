#pragma once
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include "math/BSMMath.hpp"
#include "gpu/BSMKernel.cuh"

struct PredictedGreeks {
    torch::Tensor price;
    torch::Tensor delta; // dV / dS
    torch::Tensor gamma; // d^2V / dS^2
    torch::Tensor theta; // -dV / dT (time decay)
    torch::Tensor vega;  // dV / dsigma
    torch::Tensor rho;   // dV / dr
};

class PhysicsInformedNN : public torch::nn::Module {
public:
    PhysicsInformedNN()
        // Input dimension is 5 to account for {S, K, T, r, sigma}
        : fc1(register_module("fc1", torch::nn::Linear(5, 64))),
          fc2(register_module("fc2", torch::nn::Linear(64, 64))),
          fc3(register_module("fc3", torch::nn::Linear(64, 1))) {}

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x)); // Smooth activation for 2nd derivatives
        x = torch::tanh(fc2->forward(x));
        return fc3->forward(x);
    }

    static torch::Tensor generate_targets(
        const torch::Tensor& S, const torch::Tensor& K, 
        const torch::Tensor& T, const torch::Tensor& r, 
        const torch::Tensor& sigma, const torch::Tensor& opt_type) 
    {
        int n = S.size(0);
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(S.device());
        auto target_prices = torch::empty({n, 1}, options);

    #ifdef __CUDACC__
        // GPU Path: Launch CUDA kernel for batch Black-Scholes pricing
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        launchBSMPricingTensorKernel(
            S.data_ptr<double>(), K.data_ptr<double>(), T.data_ptr<double>(),
            r.data_ptr<double>(), sigma.data_ptr<double>(), opt_type.data_ptr<int>(),
            target_prices.data_ptr<double>(), n, stream
        );
    #else
        // CPU Path: Standard CPU host loop using shared BSMMath
        auto S_a = S.accessor<double, 2>();
        auto K_a = K.accessor<double, 2>();
        auto T_a = T.accessor<double, 2>();
        auto r_a = r.accessor<double, 2>();
        auto sig_a = sigma.accessor<double, 2>();
        auto opt_a = opt_type.accessor<int, 2>();
        auto out_a = target_prices.accessor<double, 2>();

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            double d1 = BSMMath::d1(S_a[i][0], K_a[i][0], T_a[i][0], r_a[i][0], sig_a[i][0]);
            double d2 = BSMMath::d2(d1, sig_a[i][0], T_a[i][0]);
            out_a[i][0] = (opt_a[i][0] == 0) 
                ? BSMMath::callPrice(S_a[i][0], K_a[i][0], T_a[i][0], r_a[i][0], d1, d2)
                : BSMMath::putPrice(S_a[i][0], K_a[i][0], T_a[i][0], r_a[i][0], d1, d2);
            }
    #endif   
        return target_prices;

    }

    // Compute PINN Loss: Data MSE + Black-Scholes PDE Residual
    torch::Tensor compute_loss(
        torch::Tensor S, torch::Tensor K, torch::Tensor T, 
        torch::Tensor r, torch::Tensor sigma, torch::Tensor opt_type,
        torch::Tensor target_price = torch::Tensor(),
        float pde_weight = 1.0f) 
    {
        // Automatically generate ground-truth targets via CUDA kernel if not provided
        if (!target_price.defined()) {
            target_price = generate_targets(S, K, T, r, sigma, opt_type);
        }

        // Enable Autograd input tracking
        S.requires_grad_(true);
        T.requires_grad_(true);

        // Concatenate 5 features to shape (N, 5)
        auto input = torch::cat({S, K, T, r, sigma}, /*dim=*/1);
        auto V = forward(input);
        auto data_loss = torch::mse_loss(V, target_price);

        // Compute Derivatives via Autograd
        auto ones = torch::ones_like(V);
        auto grads = torch::autograd::grad({V}, {S, T}, {ones}, /*create_graph=*/true);
        auto dV_dS = grads[0];
        auto dV_dT = grads[1];

        auto d2V_dS2 = torch::autograd::grad({dV_dS}, {S}, {ones}, /*create_graph=*/true)[0];

        // Shared BSM PDE Residual evaluation
        auto pde_residual = BSMMath::pdeResidual(V, dV_dS, d2V_dS2, dV_dT, S, r, sigma);
        auto pde_loss = torch::mean(torch::pow(pde_residual, 2));

        return data_loss + (pde_weight * pde_loss);
    }

    PredictedGreeks evaluate_greeks(
        torch::Tensor S, torch::Tensor K, torch::Tensor T, 
        torch::Tensor r, torch::Tensor sigma) 
    {
        // Enable gradient tracking on input tensors
        S.requires_grad_(true);
        T.requires_grad_(true);
        r.requires_grad_(true);
        sigma.requires_grad_(true);

        auto input = torch::cat({S, K, T, r, sigma}, /*dim=*/1);
        auto V = forward(input);

        auto ones = torch::ones_like(V);

        // First-order derivatives (Delta, Theta, Rho, Vega)
        auto grads = torch::autograd::grad({V}, {S, T, r, sigma}, {ones}, /*create_graph=*/true);
        auto delta = grads[0];
        auto dV_dT = grads[1];
        auto rho   = grads[2];
        auto vega  = grads[3];

        // Second-order spatial derivative (Gamma)
        auto gamma = torch::autograd::grad({delta}, {S}, {ones}, /*create_graph=*/false)[0];

        // Time decay Theta
        auto theta = -dV_dT;

        return {V.detach(), delta.detach(), gamma.detach(), theta.detach(), vega.detach(), rho.detach()};
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};