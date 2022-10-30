#include "init_filter.h"


void init_filter(pybind11::module_& m) {
    pybind11::class_<KrylovFilter>(m, "KrylovFilter")
        .def(pybind11::init<MEInference*, ObservationModel*, const vec&, const vec&, const vec&, const vec&, const vec&>(),
            pybind11::arg("transition_model"),
            pybind11::arg("obs_model"),
            pybind11::arg("obs_times"),
            pybind11::arg("observations"),
            pybind11::arg("initial_dist"),
            pybind11::arg("rates"),
            pybind11::arg("obs_param"))
        .def("get_initial_grad", &KrylovFilter::get_initial_grad)
        .def("get_rates_grad", &KrylovFilter::get_rates_grad)
        .def("get_obs_param_grad", &KrylovFilter::get_obs_param_grad)
        .def("compute_rates_grad", &KrylovFilter::compute_rates_grad)
        .def("log_prob", &KrylovFilter::log_prob)
        .def("log_prob_backward", &KrylovFilter::log_prob_backward);
}