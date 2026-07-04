#pragma once
#include "Greeks.hpp"

class IModel {
public:
    virtual ~IModel() = default;

    // Method to calculate the price of a call option
    virtual double callPrice() const = 0;

    // Method to calculate the price of a put option
    virtual double putPrice() const = 0;

    // Method to calculate the Greeks for a call option
    virtual Greeks callGreeks() const = 0;

    // Method to calculate the Greeks for a put option
    virtual Greeks putGreeks() const = 0;
};