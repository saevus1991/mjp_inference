#include "krylov_init.h"


void krylov_init(pybind11::module_& m) {
    pybind11::class_<Krylov>(m, "Krylov")
        .def(pybind11::init<csr_mat, vec, int>(),
            pybind11::arg("generator"),
            pybind11::arg("initial"),
            pybind11::arg("order"))
        .def("get_span", &Krylov::get_span)
        .def("get_proj", &Krylov::get_proj)
        .def("expand", &Krylov::expand)
        .def("eval", &Krylov::eval);
    pybind11::class_<KrylovPropagator>(m, "KrylovPropagator")
        .def(pybind11::init<MEInference*, const vec&, const vec&, double>(),
            pybind11::arg("transition_model"),
            pybind11::arg("initial"),
            pybind11::arg("rates"),
            pybind11::arg("time"))
        .def("get_time_grad", &KrylovPropagator::get_time_grad)
        .def("get_initial_grad", &KrylovPropagator::get_initial_grad)
        .def("get_rates_grad", &KrylovPropagator::get_rates_grad)
        .def("compute_rates_grad", &KrylovPropagator::compute_rates_grad)
        .def("propagate", &KrylovPropagator::propagate)
        .def("eval", &KrylovPropagator::eval,
            pybind11::arg("time"))
        .def("backward",  static_cast<void (KrylovPropagator::*)(np_array)>(&KrylovPropagator::backward),
            pybind11::arg("grad_output"));
    pybind11::class_<KrylovSolver>(m, "KrylovSolver")
        .def(pybind11::init<MEInference*, const vec&, const vec&, const vec&>(),
            pybind11::arg("transition_model"),
            pybind11::arg("initial"),
            pybind11::arg("rates"),
            pybind11::arg("obs_times"))
        .def(pybind11::init<MEInference*, const vec&, const vec&, double>(),
            pybind11::arg("transition_model"),
            pybind11::arg("initial"),
            pybind11::arg("rates"),
            pybind11::arg("obs_times"))
        .def("get_initial_grad", &KrylovSolver::get_initial_grad)
        .def("get_rates_grad", &KrylovSolver::get_rates_grad)
        .def("compute_rates_grad", &KrylovSolver::compute_rates_grad)
        .def("forward", &KrylovSolver::forward,
            pybind11::arg("krylov_order") = 1)
        .def("backward", &KrylovSolver::backward,
            pybind11::arg("krylov_order"),
            pybind11::arg("grad_output"));
    return;
}