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
    pybind11::class_<KrylovBackwardFilter>(m, "KrylovBackwardFilter")
        .def(pybind11::init<MEInference*, ObservationModel*, vec, vec, vec, vec, vec, vec>(),
            pybind11::arg("master_equation"),
            pybind11::arg("obs_model"),
            pybind11::arg("obs_times"),
            pybind11::arg("observations"),
            pybind11::arg("initial"),
            pybind11::arg("rates"),
            pybind11::arg("obs_param"),
            pybind11::arg("tspan"))
        .def("forward_filter", &KrylovBackwardFilter::forward_filter)
        .def("eval_forward_filter", static_cast<mat_rm (KrylovBackwardFilter::*)(vec&)>(&KrylovBackwardFilter::eval_forward_filter),
            pybind11::arg("time"))
        .def("eval_forward_filter", static_cast<vec (KrylovBackwardFilter::*)(double)>(&KrylovBackwardFilter::eval_forward_filter),
            pybind11::arg("time"))
        .def("log_prob", &KrylovBackwardFilter::log_prob)
        .def("backward_filter", &KrylovBackwardFilter::backward_filter)
        .def("eval_backward_filter", static_cast<mat_rm (KrylovBackwardFilter::*)(vec&)>(&KrylovBackwardFilter::eval_backward_filter),
            pybind11::arg("time"))
        .def("eval_backward_filter", static_cast<vec (KrylovBackwardFilter::*)(double)>(&KrylovBackwardFilter::eval_backward_filter),
            pybind11::arg("time"))
        .def("eval_smoothed", static_cast<mat_rm (KrylovBackwardFilter::*)(vec&)>(&KrylovBackwardFilter::eval_smoothed),
            pybind11::arg("time"))
        .def("eval_smoothed", static_cast<vec (KrylovBackwardFilter::*)(double)>(&KrylovBackwardFilter::eval_smoothed),
            pybind11::arg("time"))
        .def("get_initial_smoothed", &KrylovBackwardFilter::get_smoothed_initial);
    m.def("batched_filter", &batched_filter,
        pybind11::arg("initial"),
        pybind11::arg("rates"),
        pybind11::arg("transition_model"),
        pybind11::arg("observation_model"),
        pybind11::arg("obs_times"),
        pybind11::arg("observations"),
        pybind11::arg("obs_param"),
        pybind11::arg("get_gradient") = false,
        pybind11::arg("num_workers") = -1,
        pybind11::arg("backend") = "krylov");
    m.def("batched_filter_list", &batched_filter_list,
        pybind11::arg("initial"),
        pybind11::arg("rates"),
        pybind11::arg("transition_model"),
        pybind11::arg("observation_model"),
        pybind11::arg("obs_times"),
        pybind11::arg("observations"),
        pybind11::arg("obs_param"),
        pybind11::arg("get_gradient") = false,
        pybind11::arg("num_workers") = -1,
        pybind11::arg("backend") = "krylov");
}