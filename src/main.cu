#include <iostream>
#include <vector>
#include "gpu/BSMKernel.cuh"
#include "gpu/BinomialTreeKernel.cuh"
#include "models/BSMModel.hpp"
#include "models/BinomialTreeModel.hpp"
#include "models/MonteCarloModel.hpp"
#include "Option.hpp"
#include "Greeks.hpp"
#include "MarketParameters.hpp"
#include <cuda_runtime.h>
#include "gpu/MonteCarloKernel.cuh"
#include "surfaces/OptionsSurface.hpp"

void BSMModelImplementation(int n_options){
    std::vector<Option> call_options(n_options, {100.0, 1.0, 0});
    std::vector<Option> put_options(n_options, {100.0, 1.0, 1});
    std::vector<MarketParams> mktparams(n_options, {100.0, 0.05, 0.2, 0.0});
    std::vector<double> call_price_results(n_options);
    std::vector<double> put_price_results(n_options);
    std::vector<Greeks> call_greek_results(n_options);
    std::vector<Greeks> put_greek_results(n_options);

    // Launch parallel computation on GPU
    launchBSMPricingKernel(call_options.data(), mktparams.data(), call_price_results.data(), n_options);
    launchBSMGreeksKernel(call_options.data(), mktparams.data(), call_greek_results.data(), n_options);

    launchBSMPricingKernel(put_options.data(), mktparams.data(), put_price_results.data(), n_options);
    launchBSMGreeksKernel(put_options.data(), mktparams.data(), put_greek_results.data(), n_options);

    std::cout << "BSM MODEL" << std::endl;
    std::cout << "Successfully processed " << n_options << " options on the GPU." << std::endl;

    std::cout << "Call Option Price + Greeks" << std::endl;
    std::cout << "Price: " << call_price_results[0] << std::endl;
    std::cout << "Delta: " << call_greek_results[0].delta << std::endl;
    std::cout << "Gamma: " << call_greek_results[0].gamma << std::endl;
    std::cout << "Vega: " << call_greek_results[0].vega << std::endl;
    std::cout << "Theta: " << call_greek_results[0].theta << std::endl;
    std::cout << "Rho: " << call_greek_results[0].rho << std::endl;

    std::cout << std::endl;

    std::cout << "Put Option Price + Greeks" << std::endl;
    std::cout << "Price: " << put_price_results[0] << std::endl;
    std::cout << "Delta: " << put_greek_results[0].delta << std::endl;
    std::cout << "Gamma: " << put_greek_results[0].gamma << std::endl;
    std::cout << "Vega: " << put_greek_results[0].vega << std::endl;
    std::cout << "Theta: " << put_greek_results[0].theta << std::endl;
    std::cout << "Rho: " << put_greek_results[0].rho << std::endl;

    std::cout << std::endl;
};

void BinomialTreeImplementation(int n_steps, int n_options){
    std::vector<Option> call_options(n_options, {100.0, 1.0, 0});
    std::vector<Option> put_options(n_options, {100.0, 1.0, 1});
    std::vector<MarketParams> mktparams(n_options, {100.0, 0.05, 0.2, 0.0});
    std::vector<double> call_price_results(n_options);
    std::vector<double> put_price_results(n_options);
    std::vector<Greeks> call_greek_results(n_options);
    std::vector<Greeks> put_greek_results(n_options);

    double delta = 0.001;

    // Launch parallel computation on GPU
    launchBinomialPricingKernel(call_options.data(), mktparams.data(), call_price_results.data(), n_steps, n_options);
    launchBinomialGreeksKernel(call_options.data(), mktparams.data(), call_greek_results.data(), delta, n_steps, n_options);

    launchBinomialPricingKernel(put_options.data(), mktparams.data(), put_price_results.data(), n_steps, n_options);
    launchBinomialGreeksKernel(put_options.data(), mktparams.data(), put_greek_results.data(), delta, n_steps, n_options);

    std::cout << "Binomial Tree MODEL" << std::endl;
    std::cout << "Successfully processed " << n_options << " options on the GPU at " << n_steps << " steps." << std::endl;

    std::cout << "Call Option Price + Greeks" << std::endl;
    std::cout << "Price: " << call_price_results[0] << std::endl;
    std::cout << "Delta: " << call_greek_results[0].delta << std::endl;
    std::cout << "Gamma: " << call_greek_results[0].gamma << std::endl;
    std::cout << "Vega: " << call_greek_results[0].vega << std::endl;
    std::cout << "Theta: " << call_greek_results[0].theta << std::endl;
    std::cout << "Rho: " << call_greek_results[0].rho << std::endl;

    std::cout << std::endl;

    std::cout << "Put Option Price + Greeks" << std::endl;
    std::cout << "Price: " << put_price_results[0] << std::endl;
    std::cout << "Delta: " << put_greek_results[0].delta << std::endl;
    std::cout << "Gamma: " << put_greek_results[0].gamma << std::endl;
    std::cout << "Vega: " << put_greek_results[0].vega << std::endl;
    std::cout << "Theta: " << put_greek_results[0].theta << std::endl;
    std::cout << "Rho: " << put_greek_results[0].rho << std::endl;

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::endl;
}

void MonteCarloModelImplementation(int n_options = 1, int n_simulations = 100000) {
    std::vector<Option> call_options(n_options, {100.0, 1.0, 0});
    std::vector<Option> put_options(n_options, {100.0, 1.0, 1});
    std::vector<MarketParams> mktparams(n_options, {100.0, 0.05, 0.2, 0.0});
    std::vector<double> call_price_results(n_options);
    std::vector<double> put_price_results(n_options);

    // Launch parallel Monte Carlo computation on GPU
    launchMonteCarloPricingKernel(call_options.data(), mktparams.data(), call_price_results.data(), n_simulations, n_options);
    launchMonteCarloPricingKernel(put_options.data(), mktparams.data(), put_price_results.data(), n_simulations, n_options);

    std::cout << "MONTE CARLO MODEL (" << n_simulations << " simulations/option)" << std::endl;
    std::cout << "Successfully processed " << n_options << " options on the GPU." << std::endl;

    std::cout << "Call Option Price" << std::endl;
    std::cout << "Price: " << call_price_results[0] << std::endl;

    std::cout << std::endl;

    std::cout << "Put Option Price" << std::endl;
    std::cout << "Price: " << put_price_results[0] << std::endl;

    std::cout << std::endl;
};

void OptionSurfaceImplementation() {
    // Grid Parameters
    int numS = 100;          
    int numVol = 100;       
    float S_min = 50.0f;    
    float S_step = 1.0f;    
    float vol_min = 0.1f;   
    float vol_step = 0.01f; 
    float T = 1.0f;         
    float r = 0.05f;        
    float K = 100.0f;       
    int optionType = 0;     

    // Ommited the GPU version that keeps results on VRAM for future operations, as it is not used in this implementation.

    // Create a vector to hold the option surface results
    std::vector<float> h_surface(numS * numVol);

    h_surface = OptionSurfaceEngine::generateGPU(
        numS, numVol, S_min, S_step, vol_min, vol_step,
        T, r, K, optionType, BSMPolicy()
    );

    std::cout << "Option Surface generated using BSM model on GPU." << std::endl;
}
int main() {
    int n_options = 1000000;
    int n_steps = 350;

    BSMModelImplementation(n_options);

    BinomialTreeImplementation(n_steps, (n_options/1000));

    MonteCarloModelImplementation(1000, 100000);

    OptionSurfaceImplementation();

    return 0;
}