#include "types.h"
#include "src/models/init_models.h"
#include "src/ssa/init_ssa.h"
#include "src/me/init_me.h"



PYBIND11_MODULE(mjp_inference, m) {
    m.doc() = "C++ implementation of a modelling software for Markov jump processes. Contains a text-based model builder, utilities for stochastic simulation, a krylov-based solver of the master equation, tools for filtering and parameter inference";
    init_models(m);
    init_ssa(m);
    init_me(m);
} 
