#pragma once

#include "../../types.h"
#include "../models/mjp.h"
#include "../util/conversion.h"


class MasterEquation {
    public:
    // constructor
    MasterEquation(MJP* mjp_, double tol_);
    MasterEquation(MJP* mjp_);

    // helpers
    std::vector<csr_mat> build_hazard_generators();
    csr_mat build_generator();

    // getters
    inline MJP* get_model() {
        return(mjp);
    }
    inline const csr_mat& get_generator() {
        return(generator);
    }
    inline const std::vector<csr_mat> get_hazard_generators() {
        return(hazard_generators);
    }

    // main functions
    np_array forward(double t, np_array_c prob);

    protected:
    MJP* mjp;
    double tol;
    std::vector<csr_mat> hazard_generators;
    csr_mat generator;

};