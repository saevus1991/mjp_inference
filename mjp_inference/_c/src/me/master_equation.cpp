#include "master_equation.h"

// constructor

MasterEquation::MasterEquation(MJP* mjp_) :
    mjp(mjp_),
    tol(1e-12)
    {
        build_base_generators();
        build_generator();
    }


// helpers

void MasterEquation::build_base_generators() {
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
    base_generators = std::vector<csr_mat>(num_events);
    for (int i = 0; i < num_events; i++) {
        base_generators[i] = csr_mat(num_states, num_states);
        base_generators[i].setFromTriplets(coefficients[i].begin(), coefficients[i].end());
    }
    return;
}

void MasterEquation::build_generator() {
    generator = base_generators[0];
    for (int i = 1; i < base_generators.size(); i++) {
        generator = generator + base_generators[i];
    }
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