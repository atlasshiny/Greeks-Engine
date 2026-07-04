#pragma once
#include "IModel.hpp"

class BSMModel : public IModel {
public:
    // Constructor
    BSMModel(double S, double K, double T, double r, double sigma);

    // Method to calculate the price of a call option
    double callPrice() const;

    // Method to calculate the price of a put option
    double putPrice() const;

    // Method to calculate the Greeks for a call option
    Greeks callGreeks() const;

    // Method to calculate the Greeks for a put option
    Greeks putGreeks() const;

private:
    double S; // Spot price
    double K; // Strike price
    double T; // Time to maturity in years
    double r; // Risk-free interest rate
    double sigma; // Volatility of the underlying asset (annualized)

    // Helper method to calculate d1
    double d1() const;

    // Helper method to calculate d2
    double d2() const;

    // Helper method to calculate greeks for both call and put options (delta, gamma, vega, theta, rho)
    // This helper specifically calculates the common Greeks for both call and put options (delta, gamma, vega, rho)
    void calculateCommonGreeks(bool isCall, double d1_val, double d2_val, double pdf_d1, Greeks& greeks) const;
    
    // This helper specifically calculates the theta for both call and put options
    void calculateCallTheta(double d1_val, double d2_val, double pdf_d1, Greeks& greeks) const;
    void calculatePutTheta(double d1_val, double d2_val, double pdf_d1, Greeks& greeks) const;

};