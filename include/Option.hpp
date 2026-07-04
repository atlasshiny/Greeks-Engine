#pragma once

enum class OptionType {
    Call,
    Put
};

struct Option {
    double S; // Spot price
    double K; // Strike price
    double T; // Time to maturity in years
    double r; // Risk-free interest rate
    double sigma; // Volatility of the underlying asset (annualized)

    OptionType type; // Option type (Call or Put)
};