#pragma once

class BSMModel {
public:
    // Constructor
    BSMModel(double S, double K, double T, double r, double sigma);

    // Method to calculate the price of a call option
    double callPrice() const;

    // Method to calculate the price of a put option
    double putPrice() const;

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
};