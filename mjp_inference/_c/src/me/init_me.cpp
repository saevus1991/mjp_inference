#include "init_me.h"


void init_me(pybind11::module_ &m){
    pybind11::class_<MasterEquation>(m, "MasterEquation")
        .def(pybind11::init<MJP*>(),
            pybind11::arg("model"))
        .def_property_readonly("generator", &MasterEquation::get_generator)
        .def("forward", &MasterEquation::forward);
    return;
}