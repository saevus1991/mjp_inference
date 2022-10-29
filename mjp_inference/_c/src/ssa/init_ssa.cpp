#include "init_ssa.h"


void init_ssa(pybind11::module_ &m){
    pybind11::class_<Simulator>(m, "Simulator")
        .def(pybind11::init<MJP*, vec, vec, int, int, std::string>(),
            pybind11::arg("model"),
            pybind11::arg("initial_state"),
            pybind11::arg("tspan"),
            pybind11::arg("seed"),
            pybind11::arg("max_events") = 100000,
            pybind11::arg("max_event_handler") = "warning")
        .def("simulate", &Simulator::simulate_py)
        .def("simulate", static_cast<np_array (Simulator::*)(np_array_c)>(&Simulator::simulate),
            pybind11::arg("t_eval"));
    m.def("simulate", static_cast<np_array (*)(np_array_c, MJP*, ObservationModel*, np_array_c, int, int, std::string)>(&simulate),
        pybind11::arg("initial_state"),
        pybind11::arg("transition_model"),
        pybind11::arg("obs_model"),
        pybind11::arg("t_eval"),
        pybind11::arg("seed"),
        pybind11::arg("max_events") = 100000,
        pybind11::arg("max_event_handler") = "warning");
    m.def("simulate", static_cast<pybind11::dict (*)(np_array_c, MJP*, np_array_c, int, int, std::string)>(&simulate_full),
        pybind11::arg("initial_state"),
        pybind11::arg("transition_model"),
        pybind11::arg("tspan"),
        pybind11::arg("seed"),
        pybind11::arg("max_events") = 100000,
        pybind11::arg("max_event_handler") = "warning");
    m.def("simulate_batched", simulate_batched,
        pybind11::arg("initial_dist"),
        pybind11::arg("rates"),
        pybind11::arg("transition_model"), 
        pybind11::arg("obs_model"),
        pybind11::arg("t_obs"),
        pybind11::arg("obs_param"),
        pybind11::arg("t_span"),
        pybind11::arg("seed"),
        pybind11::arg("num_samples") = -1,
        pybind11::arg("num_workers") = -1,
        pybind11::arg("max_events") = 100000,
        pybind11::arg("max_event_handler") = "warning");
    return;
}
