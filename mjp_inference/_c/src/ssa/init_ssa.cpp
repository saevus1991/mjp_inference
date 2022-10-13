#include "init_ssa.h"


void init_ssa(pybind11::module_ &m){
    pybind11::class_<Simulator>(m, "Simulator")
        .def(pybind11::init<MJP*, vec, vec, int, int, std::string>(),
            pybind11::arg("model"),
            pybind11::arg("initial"),
            pybind11::arg("tspan"),
            pybind11::arg("seed"),
            pybind11::arg("max_events") = 100000,
            pybind11::arg("max_event_handler") = "warning")
        .def("simulate", &Simulator::simulate_py)
        .def("simulate", static_cast<np_array (Simulator::*)(np_array_c)>(&Simulator::simulate),
            pybind11::arg("t_eval")); 
    return;
}
