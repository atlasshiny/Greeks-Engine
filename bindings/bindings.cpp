#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "Greeks.hpp"
#include "MarketParameters.hpp"
#include "Option.hpp"

#include "models/BSMModel.hpp"
#include "models/BinomialTreeModel.hpp"
#include "models/MonteCarloModel.hpp"

#include "surfaces/OptionsSurface.hpp"

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

    double* buffer = new double[n_steps + 1];

    for (int i = 0; i < n_options; ++i) {
        BinomialTreeModel model(
            ptr_mktparams[i].S, ptr_options[i].K, ptr_options[i].T, 
            ptr_mktparams[i].r, ptr_mktparams[i].sigma, ptr_mktparams[i].q, 
            n_steps, ptr_options[i].isAmerican
        );
        ptr_results[i] = model.price(ptr_options[i].type, buffer);
    }

    delete[] buffer;

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

// Templated Surface Engine Wrappers
template <typename Policy>
py::array_t<float> generate_surface_cpu(
    int numS, int numVol,
    float S_min, float S_step,
    float vol_min, float vol_step,
    float T, float r, float K, int optionType
) {
    std::vector<float> surface = OptionSurfaceEngine::generateCPU<Policy>(
        numS, numVol, S_min, S_step, vol_min, vol_step, T, r, K, optionType, Policy{}
    );

    // Shape formatted as 2D array [numS, numVol] matching std::vector<float> data
    return py::array_t<float>({numS, numVol}, surface.data());
}

#ifdef __CUDACC__
template <typename Policy>
py::array_t<float> generate_surface_gpu(
    int numS, int numVol,
    float S_min, float S_step,
    float vol_min, float vol_step,
    float T, float r, float K, int optionType
) {
    std::vector<float> h_surface = OptionSurfaceEngine<Policy>::generateGPU(
        numS, numVol,
        S_min, S_step,
        vol_min, vol_step,
        T, r, K, optionType,
        Policy{}
    );

    return py::array_t<float>({numS, numVol}, h_surface.data());
}
#endif

// Helper function to expose CPU/GPU generators per Policy type to Python
template <typename Policy>
void register_surface_generator(py::module_& m, const std::string& prefix) {
    std::string cpu_fn_name = "generate_surface_cpu_" + prefix;
    m.def(cpu_fn_name.c_str(), &generate_surface_cpu<Policy>,
          py::arg("numS"), py::arg("numVol"),
          py::arg("S_min"), py::arg("S_step"),
          py::arg("vol_min"), py::arg("vol_step"),
          py::arg("T"), py::arg("r"), py::arg("K"), py::arg("optionType"),
          "Generate option surface on CPU as a 2D NumPy float array.");

#ifdef __CUDACC__
    std::string gpu_fn_name = "generate_surface_gpu_" + prefix;
    m.def(gpu_fn_name.c_str(), &generate_surface_gpu<Policy>,
          py::arg("numS"), py::arg("numVol"),
          py::arg("S_min"), py::arg("S_step"),
          py::arg("vol_min"), py::arg("vol_step"),
          py::arg("T"), py::arg("r"), py::arg("K"), py::arg("optionType"),
          "Generate option surface on GPU as a 2D NumPy float array.");
#endif
}

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
        .def("set_parameters", &BSMModel::setParameters, 
             py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
             "Update model parameters for batch processing without creating a new instance.");

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
        .def("price", [](const MonteCarloModel& self, int optionType, std::mt19937& gen, std::normal_distribution<double>& dist) {
            return self.price(optionType, gen, dist);
        });
        .def("set_parameters", &MonteCarloModel::setParameters, 
             py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"), py::arg("numSimulations"),
             "Update model parameters for batch processing without creating a new instance.");

    // Submodule: Surfaces
    py::module_ se = m.def_submodule("surface_engine", "Surface Engine");

    // Register surface functions for each concrete policy type
    // (Replace BSMModel/BinomialTreeModel/MonteCarloModel with your exact Policy structs if they differ)
    register_surface_generator<BSMPolicy>(se, "bsm");
    register_surface_generator<BinomialTreePolicy<100>>(se, "binomial");
}