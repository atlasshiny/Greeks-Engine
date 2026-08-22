#pragma once
#include <cmath>
#include "macros.hpp"

#ifdef __CUDACC__
    #include <curand_kernel.h>
#else
    #include <random>
#endif

namespace MathUtils {
    class GBMProcess {
    public:
        HOST_DEVICE GBMProcess(double S0, double mu, double sigma) : S0(S0), mu(mu), sigma(sigma) {}

        // Method to calculate the terminal stock price
        HOST_DEVICE double stepTerminal(double T, double Z) const {
            double drift = (mu - 0.5 * sigma * sigma) * T;
            double vol = sigma * std::sqrt(T);
            return S0 * std::exp(drift + vol * Z);
        }
        
        // Method to calulate the next stock price given the current price S_current and time step dt
        HOST_DEVICE double stepNext(double S_current, double dt, double Z) const {
            double drift = (mu - 0.5 * sigma * sigma) * dt;
            double vol = sigma * std::sqrt(dt);
            return S_current * std::exp(drift + vol * Z);
        }

        #ifdef __CUDACC__
            // Device-specific sampler using cuRAND

            // sampleTerminalGPU generates a terminal stock price at time T using the GBM model on the GPU
            DEVICE double sampleTerminal(double T, curandState_t* state) const {
                double Z = curand_normal_double(state);
                return stepTerminal(T, Z);
            }

            // sampleStepGPU generates the next stock price given the current price S_current and time step dt using the GBM model on the GPU
            DEVICE double sampleStep(double S_current, double dt, curandState_t* state) const {
                double Z = curand_normal_double(state);
                return stepNext(S_current, dt, Z);
            }
        #else
            // Host-specific samplers using C++ <random>

            // sampleTerminalCPU generates a terminal stock price at time T using the GBM model
            template <typename RNG>
            double sampleTerminal(double T, RNG& gen, std::normal_distribution<double>& dist) const {
                double Z = dist(gen);
                return stepTerminal(T, Z);
            }

            // sampleStepCPU generates the next stock price given the current price S_current and time step dt using the GBM model
            template <typename RNG>
            double sampleStep(double S_current, double dt, RNG& gen, std::normal_distribution<double>& dist) const {
                double Z = dist(gen);
                return stepNext(S_current, dt, Z);
            }
        #endif

    private:
        double S0; // Initial stock price
        double mu; // Drift coefficient
        double sigma; // Volatility coefficient
    };
}