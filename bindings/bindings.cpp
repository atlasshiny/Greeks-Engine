#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Greeks.hpp"
#include "MarketParameters.hpp"
#include "Option.hpp"

#include "models/BSMModel.hpp"
#include "models/BinomialTreeModel.hpp"
#include "models/MonteCarloModel.hpp"

#ifdef __CUDACC__
    #include "gpu/BinomialTreeKernel.cuh"
    #include "gpu/BSMKernel.cuh"
    #include "gpu/MonteCarloKernel.cuh"
#endif

namespace py = pybind11;

// CPU Batch Wrappers
py::array_t<double> batch_price_binomial_cpu(py::array_t<Option> options, py::array_t<MarketParams> mktparams, int n_steps) {
    int n_options = static_cast<int>(options.request().size);
    py::array_t<double> results(n_options);

    auto* ptr_options = static_cast<Option*>(options.request().ptr);
    auto* ptr_mktparams = static_cast<MarketParams*>(mktparams.request().ptr);
    auto* ptr_results = static_cast<double*>(results.request().ptr);

    // Single buffer reused across loop iterations
    std::vector<double> buffer(n_steps + 1);

    for (int i = 0; i < n_options; ++i) {
        BinomialTreeModel model(
            ptr_mktparams[i].S, ptr_options[i].K, ptr_options[i].T, 
            ptr_mktparams[i].r, ptr_mktparams[i].sigma, ptr_mktparams[i].q, 
            n_steps, buffer.data(), ptr_options[i].isAmerican
        );
        ptr_results[i] = model.price(ptr_options[i].type);
    }

    return results;
}

// CUDA Batch Wrappers
#ifdef __CUDACC__
py::array_t<double> batch_price_binomial_gpu(py::array_t<Option> options, py::array_t<MarketParams> mktparams, int n_steps) {
    int n_options = static_cast<int>(options.request().size);
    py::array_t<double> results(n_options);

    launchBinomialPricingKernel(
        static_cast<Option*>(options.request().ptr), 
        static_cast<MarketParams*>(mktparams.request().ptr), 
        static_cast<double*>(results.request().ptr), 
        n_steps, 
        n_options
    );

    return results;
}

py::array_t<double> batch_price_bsm_gpu(py::array_t<Option> options, py::array_t<MarketParams> mktparams) {
    int n_options = static_cast<int>(options.request().size);
    py::array_t<double> results(n_options);

    launchBSMPricingKernel(
        static_cast<Option*>(options.request().ptr), 
        static_cast<MarketParams*>(mktparams.request().ptr), 
        static_cast<double*>(results.request().ptr), 
        n_options
    );

    return results;
}

py::array_t<double> batch_price_monte_carlo_gpu(py::array_t<Option> options, py::array_t<MarketParams> mktparams, int numSimulations) {
    int n_options = static_cast<int>(options.request().size);
    py::array_t<double> results(n_options);

    launchMonteCarloPricingKernel(
        static_cast<Option*>(options.request().ptr), 
        static_cast<MarketParams*>(mktparams.request().ptr), 
        static_cast<double*>(results.request().ptr), 
        numSimulations, 
        n_options
    );

    return results;
}
#endif

// Pybind11 Engine Module Definition
PYBIND11_MODULE(GreeksEngine, m) {
    m.doc() = "High-performance Option Pricing and Greeks Engine (CPU & GPU)";

    // Bind common data structures required for batch inputs
    py::class_<Option>(m, "Option")
        .def(py::init<int, double, double, bool>(), 
             py::arg("type"), py::arg("K"), py::arg("T"), py::arg("isAmerican") = false)
        .def_readwrite("type", &Option::type)
        .def_readwrite("K", &Option::K)
        .def_readwrite("T", &Option::T)
        .def_readwrite("isAmerican", &Option::isAmerican);

    py::class_<MarketParams>(m, "MarketParams")
        .def(py::init<double, double, double, double>(), 
             py::arg("S"), py::arg("r"), py::arg("sigma"), py::arg("q") = 0.0)
        .def_readwrite("S", &MarketParams::S)
        .def_readwrite("r", &MarketParams::r)
        .def_readwrite("sigma", &MarketParams::sigma)
        .def_readwrite("q", &MarketParams::q);

    py::class_<Greeks>(m, "Greeks")
        .def_readonly("delta", &Greeks::delta)
        .def_readonly("gamma", &Greeks::gamma)
        .def_readonly("vega", &Greeks::vega)
        .def_readonly("theta", &Greeks::theta)
        .def_readonly("rho", &Greeks::rho);

    // Submodule: BSM
    py::module_ bsm = m.def_submodule("bsm", "Black-Scholes-Merton Pricing Module");
    py::class_<BSMModel>(bsm, "BSMModel")
        .def(py::init<double, double, double, double, double>(),
             py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"))
        .def("price", &BSMModel::price, py::arg("optionType"))
        .def("calculate_greeks", &BSMModel::calculateGreeks, py::arg("optionType"));

    // Submodule: Binomial
    py::module_ binomial = m.def_submodule("binomial", "Binomial Tree Pricing Module");
    
    binomial.def("batch_price_cpu", &batch_price_binomial_cpu, 
                 py::arg("options"), py::arg("market_params"), py::arg("n_steps"),
                 "Batch price options on CPU without exposed buffer allocations.");

#ifdef __CUDACC__
    binomial.def("batch_price_gpu", &batch_price_binomial_gpu, 
                 py::arg("options"), py::arg("market_params"), py::arg("n_steps"),
                 "Batch price options on GPU using CUDA.");
#endif

    // Submodule: Monte Carlo
    py::module_ mc = m.def_submodule("monte_carlo", "Monte Carlo Pricing Module");
    py::class_<MonteCarloModel>(mc, "MonteCarloModel")
        .def(py::init<double, double, double, double, double, int>(),
             py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"), py::arg("numSimulations"))
        .def("price", &MonteCarloModel::price, py::arg("optionType"));
}