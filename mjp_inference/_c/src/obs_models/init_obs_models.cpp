#include "init_obs_models.h"


void init_obs_models(pybind11::module_ &m){
    pybind11::class_<Param>(m, "Param")
        .def(pybind11::init<std::string, const vec&>(),
            pybind11::arg("name"),
            pybind11::arg("value"))
        .def_property("name", &Param::get_name, &Param::set_name)
        .def_property_readonly("value", &Param::get_value);
    pybind11::class_<Transform>(m, "Transform")
        .def(pybind11::init<const std::string&, pybind11::tuple, unsigned, pybind11::tuple>(),
            pybind11::arg("name"),
            pybind11::arg("transform_callable"),
            pybind11::arg("output_dim") = 1,
            pybind11::arg("grad_callable") = pybind11::tuple())
        .def_property_readonly("name", &Transform::get_name)
        .def_property_readonly("output_dim", &Transform::get_output_dim)
        .def("transform", &Transform::transform,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"))
        .def("grad_param", &Transform::grad,
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
        .def("log_prob", static_cast<double (NoiseModel::*)(const vec&)>(&NoiseModel::log_prob), 
            pybind11::arg("obs"))
        .def("log_prob_grad", static_cast<std::vector<vec> (NoiseModel::*)(const vec&)>(&NoiseModel::log_prob_grad), 
            pybind11::arg("obs"));
    pybind11::class_<Normal, NoiseModel>(m, "NormalNoise")
        .def(pybind11::init<const vec&, const vec&>(),
            pybind11::arg("mu"),
            pybind11::arg("sigma"));
        pybind11::class_<LogNormal, Normal>(m, "LogNormalNoise")
        .def(pybind11::init<const vec&, const vec&>(),
            pybind11::arg("exp_mu"),
            pybind11::arg("sigma"));
    pybind11::class_<ObservationModel>(m, "ObservationModel")
        .def(pybind11::init<MJP*, const std::string&>(),
            pybind11::arg("transition_model"),
            pybind11::arg("noise_type"))
        .def_property_readonly("noise_type", &ObservationModel::get_noise_type)
        .def_property_readonly("noise_param_list", &ObservationModel::get_noise_param_list)
        .def_property_readonly("num_param", &ObservationModel::get_num_param)
        .def_property_readonly("obs_dim", &ObservationModel::get_obs_dim)
        .def_property_readonly("param_list", &ObservationModel::get_param_list)
        .def_property_readonly("param_map", &ObservationModel::get_param_map)
        .def_property_readonly("param_array", &ObservationModel::get_param_array)
        .def_property_readonly("param_parser", &ObservationModel::get_param_parser)
        .def("build", &ObservationModel::build)
        .def("add_param", &ObservationModel::add_param,
            pybind11::arg("param"))
        // .def("add_param", static_cast <void (ObservationModel::*)(const std::string&, const vec&)>(&ObservationModel::make_add_param),
        //     pybind11::arg("name"),
        //     pybind11::arg("value"))
        .def("add_param", static_cast <void (ObservationModel::*)(const std::string&, double)>(&ObservationModel::make_add_param),
            pybind11::arg("name"),
            pybind11::arg("value"))
        .def("add_transform", &ObservationModel::add_transform,
            pybind11::arg("transform"))
        .def("log_prob", &ObservationModel::log_prob,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"),
            pybind11::arg("obs"))
        .def("log_prob_grad", &ObservationModel::log_prob_grad,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"),
            pybind11::arg("obs"))
        .def("sample", &ObservationModel::sample_np,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"),
            pybind11::arg("seed"))
        .def("transform", &ObservationModel::transform,
            pybind11::arg("time"),
            pybind11::arg("state"),
            pybind11::arg("param"),
            pybind11::arg("name"))
        .def("log_prob_vec", &ObservationModel::log_prob_vec,
            pybind11::arg("time"),
            pybind11::arg("param"),
            pybind11::arg("obs"))
        .def("log_prob_grad_vec", &ObservationModel::log_prob_grad_vec,
            pybind11::arg("time"),
            pybind11::arg("param"),
            pybind11::arg("obs"))
        .def("transform_vec", &ObservationModel::transform_vec,
            pybind11::arg("time"),
            pybind11::arg("param"),
            pybind11::arg("name"));
    return;
}