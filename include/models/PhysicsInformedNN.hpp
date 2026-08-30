#pragma once
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include "math/BSMMath.hpp"

#ifdef USE_CUDA
    #include <cuda_runtime.h>
    #include "gpu/BSMKernel.cuh"
#endif

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
          fc3(register_module("fc3", torch::nn::Linear(64, 1))) {
            // Ensure the model operates in double precision
            fc1->to(torch::kFloat64);
            fc2->to(torch::kFloat64);
            fc3->to(torch::kFloat64);
          }

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

        // ONLY execute custom CUDA kernel if input tensors reside on GPU
        if (S.is_cuda()) {
            
        #ifdef USE_CUDA
            cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
            launchBSMPricingTensorKernel(
                S.data_ptr<double>(), K.data_ptr<double>(), T.data_ptr<double>(),
                r.data_ptr<double>(), sigma.data_ptr<double>(), opt_type.data_ptr<int>(),
                target_prices.data_ptr<double>(), n, stream
            );
        #else
            TORCH_CHECK(false, "Tensor is on CUDA, but project compiled without CUDA support!");
        #endif

        } else {
            // CPU fallback execution
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
        }   
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

        // Concatenate raw inputs, then make the concatenated block track gradients
        auto input = torch::cat({S, K, T, r, sigma}, /*dim=*/1);
        input.requires_grad_(true);

        auto V = forward(input);
        auto data_loss = torch::mse_loss(V, target_price);

        auto ones = torch::ones_like(V);
        
        // 1. First-order derivative w.r.t input
        auto grads = torch::autograd::grad({V}, {input}, {ones}, 
                                        /*create_graph=*/true, 
                                        /*retain_graph=*/true, 
                                        /*allow_unused=*/true)[0];

        // Slice out dV/dS (col 0) and dV/dT (col 2)
        auto dV_dS = grads.slice(/*dim=*/1, /*start=*/0, /*end=*/1);
        auto dV_dT = grads.slice(/*dim=*/1, /*start=*/2, /*end=*/3);

        // 2. Second-order derivative d2V/dS2
        auto d2_grads = torch::autograd::grad({dV_dS}, {input}, {ones}, 
                                            /*create_graph=*/true, 
                                            /*retain_graph=*/true, 
                                            /*allow_unused=*/true)[0];

        auto d2V_dS2 = d2_grads.defined() 
            ? d2_grads.slice(/*dim=*/1, /*start=*/0, /*end=*/1) 
            : torch::zeros_like(S);

        auto pde_residual = BSMMath::pdeResidual(V, dV_dS, d2V_dS2, dV_dT, S, r, sigma);
        auto pde_loss = torch::mean(torch::pow(pde_residual, 2));

        return data_loss + (pde_weight * pde_loss);
    }
    
    // Evaluate the Greeks (Delta, Gamma, Theta, Vega, Rho) for a given set of inputs
    PredictedGreeks evaluate_greeks(
        torch::Tensor S, torch::Tensor K, torch::Tensor T, 
        torch::Tensor r, torch::Tensor sigma) 
    {
        auto input = torch::cat({S, K, T, r, sigma}, /*dim=*/1);
        input.requires_grad_(true);

        auto V = forward(input);

        auto ones = torch::ones_like(V);

        auto grads = torch::autograd::grad({V}, {input}, {ones}, 
                                        /*create_graph=*/true, 
                                        /*retain_graph=*/true, 
                                        /*allow_unused=*/true)[0];

        auto delta = grads.slice(/*dim=*/1, 0, 1);
        auto dV_dT = grads.slice(/*dim=*/1, 2, 3);
        auto rho   = grads.slice(/*dim=*/1, 3, 4);
        auto vega  = grads.slice(/*dim=*/1, 4, 5);

        auto d2_grads = torch::autograd::grad({delta}, {input}, {ones}, 
                                            /*create_graph=*/false, 
                                            /*retain_graph=*/true, 
                                            /*allow_unused=*/true)[0];

        auto gamma = d2_grads.defined() 
            ? d2_grads.slice(/*dim=*/1, 0, 1) 
            : torch::zeros_like(S);

        auto theta = -dV_dT;

        return {V.detach(), delta.detach(), gamma.detach(), theta.detach(), vega.detach(), rho.detach()};
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};