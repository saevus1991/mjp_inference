#pragma once

#include "master_equation.h"


class MEInference : public MasterEquation {
    public:

    //constructor
    MEInference(MJP* mjp_);

    // helpers
    void build_param_generators();

    // getters
    inline const std::vector<csr_mat>& get_param_generators() {
        return(param_generators);
    }


    private:
    std::vector<csr_mat> param_generators;
};