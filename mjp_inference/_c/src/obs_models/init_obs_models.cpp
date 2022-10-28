#include "init_obs_models.h"


void init_obs_models(pybind11::module_ &m){
    pybind11::class_<Param>(m, "Param")
        .def(pybind11::init<std::string, const vec&>(),
            pybind11::arg("name"),
            pybind11::arg("value"))
        .def_property("name", &Param::get_name, &Param::set_name)
        .def_property_readonly("value", &Param::get_value);
    pybind11::class_<Transform>(m, "Transform")
        .def(pybind11::init<const std::string&, pybind11::tuple, unsigned, pybind11::tuple, pybind11::tuple>(),
            pybind11::arg("name"),
            pybind11::arg("transform_callable"),
            pybind11::arg("output_dim") = 1,
            pybind11::arg("grad_state_callable") = pybind11::tuple(),
            pybind11::arg("grad_param_callable") = pybind11::tuple())
        .def_property_readonly("name", &Transform::get_name)
        .def_property_readonly("output_dim", &Transform::get_output_dim)
        .def("transform", &Transform::transform,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"))
        .def("grad_state", &Transform::grad_state,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"),
            pybind11::arg("grad_output"))
        .def("grad_param", &Transform::grad_param,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"),
            pybind11::arg("grad_output"));
    pybind11::class_<NoiseModel, PyNoiseModel>(m, "NoiseModel")
        .def(pybind11::init<const std::vector<Param>&>(),
            pybind11::arg("param_list"))
        .def_property_readonly("param_list", &NoiseModel::get_param_list)
        .def("sample", static_cast<vec (NoiseModel::*)(unsigned)>(&NoiseModel::sample),
            pybind11::arg("seed") = std::random_device()())
        .def("log_prob", &NoiseModel::log_prob, 
            pybind11::arg("obs"))
        .def("log_prob_grad", &NoiseModel::log_prob_grad, 
            pybind11::arg("obs"));
    pybind11::class_<Normal, NoiseModel>(m, "NormalNoise")
        .def(pybind11::init<const vec&, const vec&>(),
            pybind11::arg("mu"),
            pybind11::arg("sigma"));
    return;
}