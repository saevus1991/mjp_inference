#include "init_me.h"


void init_me(pybind11::module_ &m){
    pybind11::class_<MasterEquation>(m, "MasterEquation")
        .def(pybind11::init<MJP*>(),
            pybind11::arg("model"))
        .def_property_readonly("generator", &MasterEquation::get_generator)
        .def_property_readonly("hazard_generators", &MasterEquation::get_hazard_generators)
        .def_property_readonly("model", &MasterEquation::get_model)
        .def("forward", &MasterEquation::forward,
            pybind11::arg("time"),
            pybind11::arg("prob"));
    pybind11::class_<MEInference, MasterEquation>(m, "MEInference")
        .def(pybind11::init<MJP*>(),
            pybind11::arg("model"))
        .def_property_readonly("propensity_generators", &MEInference::get_propensity_generators)
        .def_property_readonly("param_generators", &MEInference::get_param_generators)
        .def("update_generator", &MEInference::update_generator,
            pybind11::arg("rates"))
        .def("forward", &MEInference::forward,
            pybind11::arg("time"),
            pybind11::arg("prob"),
            pybind11::arg("rates"))
        .def("forward", &MEInference::augmented_backward,
            pybind11::arg("time"),
            pybind11::arg("backward"),
            pybind11::arg("forward"),
            pybind11::arg("rates"));
    return;
}