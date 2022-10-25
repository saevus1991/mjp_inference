#pragma once

#include "master_equation.h"


class MEInference : public MasterEquation {
    public:

    //constructor
    MEInference(MJP* mjp_);

    // helpers
    std::vector<csr_mat> build_propensity_generators();
    std::vector<csr_mat> build_param_generators();
    csr_mat build_generator();

    // getters
    inline const std::vector<csr_mat>& get_param_generators() {
        return(param_generators);
    }
    inline const std::vector<csr_mat>& get_propensity_generators() {
        return(propensity_generators);
    }
    inline csr_mat get_generator(const vec& rates) const{
        // recompute the generator
        csr_mat generator_ = rates[0] * param_generators[0];
        for (int i = 1; i < rates.rows(); i++) {
            generator_ += rates[i] * param_generators[i];
        }
        return(generator_);
    }


    private:
    std::vector<csr_mat> propensity_generators;
    std::vector<csr_mat> param_generators;
};