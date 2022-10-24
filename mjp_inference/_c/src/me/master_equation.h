#pragma once

#include "../../types.h"
#include "../models/mjp.h"
#include "../util/conversion.h"


class MasterEquation {
    public:
    // constructor
    MasterEquation(MJP* mjp_);

    // helpers
    void build_base_generators();
    void build_generator();

    // getters
    inline const csr_mat& get_generator() {
        return(generator);
    }
    inline const std::vector<csr_mat> get_base_generators() {
        return(base_generators);
    }

    // main functions
    np_array forward(double t, np_array_c prob);

    protected:
    MJP* mjp;
    double tol;
    std::vector<csr_mat> base_generators;
    csr_mat generator;

};