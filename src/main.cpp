#include <iostream>
#include <random>
#include <cmath>
#include "models/BinomialTreeModel.hpp"
#include "models/BSMModel.hpp"
#include "models/MonteCarloModel.hpp"
#include "Greeks.hpp"
#include "surfaces/OptionsSurface.hpp"

void BSMModelImplementation() {
    // Example parameters for the BSM model
    double S = 100.0;   // Spot price
    double K = 100.0;   // Strike price
    double T = 1.0;     // Time to maturity in years
    double r = 0.05;    // Risk-free interest rate
    double sigma = 0.2; // Volatility of the underlying asset

    // Create an instance of the BSMModel
    BSMModel bsm(S, K, T, r, sigma);

    // Calculate and display the call and put option prices
    double callPrice = bsm.price(0); // 0 for call option
    Greeks callGreeks = bsm.calculateGreeks(0); // 0 for call option
    double putPrice = bsm.price(1); // 1 for put option
    Greeks putGreeks = bsm.calculateGreeks(1); // 1 for put option

    std::cout << "Call Option Price: " << callPrice << std::endl;
    std::cout << "Call Option Greeks:" << std::endl;
    std::cout << "Delta: " << callGreeks.delta << std::endl;
    std::cout << "Gamma: " << callGreeks.gamma << std::endl;
    std::cout << "Vega: " << callGreeks.vega << std::endl;
    std::cout << "Theta: " << callGreeks.theta << std::endl;
    std::cout << "Rho: " << callGreeks.rho << std::endl;

    std::cout << std::endl;

    std::cout << "Put Option Price: " << putPrice << std::endl;
    std::cout << "Put Option Greeks:" << std::endl;
    std::cout << "Delta: " << putGreeks.delta << std::endl;
    std::cout << "Gamma: " << putGreeks.gamma << std::endl;
    std::cout << "Vega: " << putGreeks.vega << std::endl;
    std::cout << "Theta: " << putGreeks.theta << std::endl;
    std::cout << "Rho: " << putGreeks.rho << std::endl;

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::endl;
}

void BinomialTreeImplementation() {
    // Example parameters for the Binomial Tree model
    double S = 100.0;        // Spot price
    double K = 100.0;        // Strike price
    double T = 1.0;          // Time to maturity in years
    double r = 0.05;         // Risk-free interest rate
    double sigma = 0.2;      // Volatility of the underlying asset
    double q = 0.0;          // Dividend yield
    int steps = 1000;        // Number of steps in the binomial tree
    bool isAmerican = false; // Flag to indicate if the option is American or European

    // Create an instance of the BinomialTreeModel
    BinomialTreeModel binomial(S, K, T, r, sigma, q, steps, isAmerican);

    // Allocate a buffer for option values (size N+1)
    std::vector<double> buffer(steps + 1);

    // Calculate and display the call and put option prices
    double callPrice = binomial.price(0, buffer.data()); // 0 for call option
    Greeks callGreeks = binomial.calculateGreeks(0, buffer.data(), 0.01); // 0 for call option, h=0.01 for finite difference
    double putPrice = binomial.price(1, buffer.data());  // 1 for put option
    Greeks putGreeks = binomial.calculateGreeks(1, buffer.data(), 0.01); // 1 for put option, h=0.01 for finite difference

    std::cout << "Binomial Tree Model:" << std::endl;
    std::cout << "Call Option Price: " << callPrice << std::endl;
    std::cout << "Delta: " << callGreeks.delta << std::endl;
    std::cout << "Gamma: " << callGreeks.gamma << std::endl;
    std::cout << "Vega: " << callGreeks.vega << std::endl;
    std::cout << "Theta: " << callGreeks.theta << std::endl;
    std::cout << "Rho: " << callGreeks.rho << std::endl;

    std::cout << std::endl;

    std::cout << "Put Option Price: " << putPrice << std::endl;
    std::cout << "Put Option Greeks:" << std::endl;
    std::cout << "Delta: " << putGreeks.delta << std::endl;
    std::cout << "Gamma: " << putGreeks.gamma << std::endl;
    std::cout << "Vega: " << putGreeks.vega << std::endl;
    std::cout << "Theta: " << putGreeks.theta << std::endl;
    std::cout << "Rho: " << putGreeks.rho << std::endl;

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::endl;

    // When Greek calculations are implemented, call the calculateGreeks method similarly to the BSM model.
}

void MonteCarloImplementation(int n_options, int n_simulations) {
    // Example parameters for the Monte Carlo model
    double S = 100.0;        // Spot price
    double K = 100.0;        // Strike price
    double T = 1.0;          // Time to maturity in years
    double r = 0.05;         // Risk-free interest rate
    double sigma = 0.2;      // Volatility of the underlying asset
    int numSimulations = n_simulations; // Number of Monte Carlo simulations

    // Statistical parameters to pass into the Monte Carlo model for sampling
    std::random_device rd;  // Random number generator
    std::normal_distribution<double> dist(0.0, 1.0); // Standard normal distribution

    for (int i = 0; i < n_options; ++i) {
        // Create an instance of the MonteCarloModel
        MonteCarloModel monteCarlo(S, K, T, r, sigma, numSimulations);

        // Calculate and display the call and put option prices
        double callPrice = monteCarlo.price(0, rd, dist); // 0 for call option
        double putPrice = monteCarlo.price(1, rd, dist);  // 1 for put option

        std::cout << "Monte Carlo Model (Option " << i + 1 << "):" << std::endl;
        std::cout << "Call Option Price: " << callPrice << std::endl;
        std::cout << "Put Option Price: " << putPrice << std::endl;
        std::cout << std::endl;
    }

    // Note: Greeks calculation is not implemented for Monte Carlo due to computational expense.
}

void OptionSurfaceImplementation(){
    // Example parameters for the Option Surface generation
    int numS = 100;          // Number of spot price points
    int numVol = 100;       // Number of volatility points
    float S_min = 50.0f;    // Minimum spot price
    float S_step = 1.0f;    // Step size for spot price
    float vol_min = 0.1f;   // Minimum volatility
    float vol_step = 0.01f; // Step size for volatility
    float T = 1.0f;         // Time to maturity in years
    float r = 0.05f;        // Risk-free interest rate
    float K = 100.0f;       // Strike price
    int optionType = 0;     // Option type: 0 for Call, 1 for Put

    // Allocate a buffer for the option surface (size numS * numVol)
    std::vector<float> surface(numS * numVol);

    // Generate the option surface using the BSM pricing policy on CPU
    OptionSurfaceEngine::generateCPU(
        surface.data(),
        numS, numVol,
        S_min, S_step,
        vol_min, vol_step,
        T, r, K, optionType,
        BSMPolicy()
    );

    std::cout << "Option Surface generated using BSM model on CPU." << std::endl;
}

// Runs every model that has been created with default parameters and prints the results to the console
int main() {
    BSMModelImplementation();

    BinomialTreeImplementation();

    MonteCarloImplementation(1000, 100000);

    OptionSurfaceImplementation();

    return 0;
}