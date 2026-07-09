#include <iostream>
#include "models/BinomialTreeModel.hpp"
#include "models/BSMModel.hpp"
#include "Greeks.hpp"

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

    // When Greek calculations are implemented, call the calculateGreeks method similarly to the BSM model.
}

// Runs every model that has been created with default parameters and prints the results to the console
int main() {
    BSMModelImplementation();

    BinomialTreeImplementation();

    return 0;
}