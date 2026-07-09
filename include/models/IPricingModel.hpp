#pragma once
#include "Greeks.hpp"
#include "macros.hpp"

class IPricingModel {
public:
    HOST_DEVICE virtual ~IPricingModel() = default;

    // Method to calculate the price of an option
    HOST_DEVICE virtual double price(int optionType) const = 0;

    // Method to calculate the Greeks of an option
    HOST_DEVICE virtual Greeks calculateGreeks(int optionType) const = 0;
};