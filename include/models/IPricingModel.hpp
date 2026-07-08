#pragma once
#include "Greeks.hpp"

class IPricingModel {
public:
    virtual ~IPricingModel() = default;

    // Method to calculate the price of an option
    virtual double price(int optionType) const = 0;

    // Method to calculate the Greeks of an option
    virtual Greeks calculateGreeks(int optionType) const = 0;
};