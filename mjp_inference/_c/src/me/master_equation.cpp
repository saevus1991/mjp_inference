#include "master_equation.h"

// constructor

MasterEquation::MasterEquation(MJP* mjp_, double tol_) :
    mjp(mjp_),
    tol(tol_)
    {}

MasterEquation::MasterEquation(MJP* mjp_) :
    MasterEquation(mjp_, 1e-12)
    {
        hazard_generators = build_hazard_generators();
        generator = build_generator();
    }


// helpers

std::vector<csr_mat> MasterEquation::build_hazard_generators() {
    // preparations
    unsigned num_events = mjp->get_num_events();
    unsigned num_states = mjp->get_num_states();
    std::vector<int> dims(mjp->get_dims());
    // create transition triplets
    std::vector<std::vector<triplet>> coefficients(num_events);
    for (int i = 0; i < num_states; i++) {
        // convert to state
        std::vector<double> state = ut::lin2state<double, int>(i, dims);
        // compute hazard
        std::vector<double> hazard(num_events);
        mjp->hazard(state.data(), hazard.data());
        std::vector<double> exit_rates(num_events);
        for (int j = 0; j < hazard.size(); j++) {
            if ( std::abs(hazard[j]) > tol ) {
                std::vector<double> target(state);
                mjp->update_state(target.data(), j);
                if ( mjp->is_valid_state(target) ) {
                    int k = ut::state2lin(target, dims);
                    coefficients[j].push_back(triplet(i, k, hazard[j]));
                    coefficients[j].push_back(triplet(i, i, -hazard[j]));
                }
            }
        }
    }
    // set up generators from triplet
    std::vector<csr_mat> hazard_generators(num_events);
    for (int i = 0; i < num_events; i++) {
        hazard_generators[i] = csr_mat(num_states, num_states);
        hazard_generators[i].setFromTriplets(coefficients[i].begin(), coefficients[i].end());
    }
    return(hazard_generators);
}

csr_mat MasterEquation::build_generator() {
    csr_mat generator(mjp->get_num_states(), mjp->get_num_states());
    for (int i = 0; i < hazard_generators.size(); i++) {
        generator = generator + hazard_generators[i];
    }
    return(generator);
}

// main functions

np_array MasterEquation::forward(double t, np_array_c prob_) {
    // parse input
    Eigen::Map<vec> prob((double*) prob_.data(), prob_.size());
    // set up output
    np_array dpdt_(prob_.size());
    Eigen::Map<vec> dpdt((double*)dpdt_.data(), dpdt_.size());
    // compute and return
    dpdt.noalias() = generator.transpose() * prob;
    return(dpdt_);
}