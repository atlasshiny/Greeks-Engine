#include <pybind11/pybind11.h>
#include "models/BSMModel.hpp"
#include "models/BinomialTreeModel.hpp"

namespace py = pybind11;

PYBIND11_MODULE(GreeksEngine, m) {
    m.doc() = "pybind11 bindings for the CXX/CUDA Greeks Engine";
    
    // Define all of the models in /include/models here
    // Currently only BSMModel is binded since it doesn't have buffers in the method signatures
    py::class_<BSMModel>(m, "BSMModel")
        .def(py::init<double, double, double, double, double>(),
             py::arg("S"), py::arg("K"), py::arg("T"),
             py::arg("r"), py::arg("sigma"))
        .def("price", &BSMModel::price, py::arg("optionType"))
        .def("calculate_greeks", &BSMModel::calculateGreeks,
             py::arg("optionType"));

}