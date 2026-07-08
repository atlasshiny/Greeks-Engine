#include <iostream>

#include "models/BSMModel.hpp"

int main() {
    // Example parameters for the BSM model
    double S;     // Spot price
    double K;     // Strike price
    double T;     // Time to maturity in years
    double r;     // Risk-free interest rate
    double sigma; // Volatility of the underlying asset

    // Get user input for the parameters
    std::cout << "Enter Spot Price (S): ";
    std::cin >> S;
    std::cout << "Enter Strike Price (K): ";
    std::cin >> K;
    std::cout << "Enter Time to Maturity (T in years): ";
    std::cin >> T;
    std::cout << "Enter Risk-Free Interest Rate (r): ";
    std::cin >> r;
    std::cout << "Enter Volatility (sigma): ";
    std::cin >> sigma;

    // Create an instance of the BSMModel
    BSMModel bsm(S, K, T, r, sigma);

    // Calculate and display the call and put option prices
    double callPrice = bsm.price(0); // 0 for call option
    double putPrice = bsm.price(1); // 1 for put option

    std::cout << "Call Option Price: " << callPrice << std::endl;
    std::cout << "Put Option Price: " << putPrice << std::endl;

    return 0;
}