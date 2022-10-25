#include "init_me.h"


void init_me(pybind11::module_ &m){
    pybind11::class_<MasterEquation>(m, "MasterEquation")
        .def(pybind11::init<MJP*>(),
            pybind11::arg("model"))
        .def_property_readonly("generator", &MasterEquation::get_generator)
        .def_property_readonly("hazard_generators", &MasterEquation::get_hazard_generators)
        .def("forward", &MasterEquation::forward);
    pybind11::class_<MEInference, MasterEquation>(m, "MEInference")
        .def(pybind11::init<MJP*>(),
            pybind11::arg("model"))
        .def_property_readonly("propensity_generators", &MEInference::get_propensity_generators)
        .def_property_readonly("param_generators", &MEInference::get_param_generators);
    return;
}