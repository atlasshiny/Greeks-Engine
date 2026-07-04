#include <iostream>

#include "BSMModel.hpp"

int main() {
    // Example parameters for the BSM model
    double S = 100.0;   // Spot price
    double K = 100.0;   // Strike price
    double T = 1.0;     // Time to maturity in years
    double r = 0.05;    // Risk-free interest rate
    double sigma = 0.2; // Volatility of the underlying asset

    // Create an instance of the BSMModel
    BSMModel bsm(S, K, T, r, sigma);

    // Calculate and display the call and put option prices
    double callPrice = bsm.callPrice();
    double putPrice = bsm.putPrice();

    std::cout << "Call Option Price: " << callPrice << std::endl;
    std::cout << "Put Option Price: " << putPrice << std::endl;

    return 0;
}