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
    return;
}