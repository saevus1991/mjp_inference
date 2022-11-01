#include "me_inference.h"

// constructor

MEInference::MEInference(MJP* mjp_) : 
    MasterEquation(mjp_, 1e-12),
    propensity_generators(build_propensity_generators()),
    param_generators(build_param_generators())
    {
        generator = build_generator();
    }


// helpers

std::vector<csr_mat> MEInference::build_propensity_generators() {
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
        std::vector<double> propensity(num_events);
        mjp->propensity(state.data(), propensity.data());
        std::vector<double> exit_rates(num_events);
        for (int j = 0; j < propensity.size(); j++) {
            if ( std::abs(propensity[j]) > tol ) {
                std::vector<double> target(state);
                mjp->update_state(target.data(), j);
                if ( mjp->is_valid_state(target) ) {
                    int k = ut::state2lin(target, dims);
                    coefficients[j].push_back(triplet(i, k, propensity[j]));
                    coefficients[j].push_back(triplet(i, i, -propensity[j]));
                }
            }
        }
    }
    // set up generators from triplet
    std::vector<csr_mat> propensity_generators(num_events);
    for (int i = 0; i < num_events; i++) {
        propensity_generators[i] = csr_mat(num_states, num_states);
        propensity_generators[i].setFromTriplets(coefficients[i].begin(), coefficients[i].end());
    }
    return(propensity_generators);
}

std::vector<csr_mat> MEInference::build_param_generators() {
    // set up genrators
    std::vector<csr_mat> param_generators;
    unsigned num_rates = mjp->get_num_rates();
    for (unsigned i = 0; i < num_rates; i++) {
        param_generators.push_back(csr_mat(mjp->get_num_states(), mjp->get_num_states()));
    }
    // add contributions
    unsigned num_events = mjp->get_num_events();
    for (unsigned i = 0; i < num_events; i++) {
        unsigned rate_index = mjp->event2rate(i);
        param_generators[rate_index] = param_generators[rate_index] + propensity_generators[i];
    }
    return(param_generators);
}

csr_mat MEInference::build_generator(const vec& rates) {
    // get rates
    csr_mat generator(mjp->get_num_states(), mjp->get_num_states());
    for (unsigned i = 0; i < mjp->get_num_rates(); i++) {
        generator = generator + rates[i] * param_generators[i];
    }
    return(generator);
}

csr_mat MEInference::build_generator() {
    vec rates = mjp->get_rate_array();
    return(build_generator(rates));
} 

void MEInference::update_generator(const vec& rates) {
    generator = build_generator(rates);
}

// main functions

np_array MEInference::forward(double t, np_array_c prob_, np_array_c rates_) {
    // parse input
    Eigen::Map<vec> prob((double*) prob_.data(), prob_.size());
    Eigen::Map<vec> rates((double*) rates_.data(), rates_.size());
    // set up output
    np_array dpdt_(prob_.size());
    Eigen::Map<vec> dpdt((double*)dpdt_.data(), dpdt_.size());
    // compute and return
    dpdt.noalias() = param_generators[0].transpose() * prob;
    for (unsigned i =1; i < rates.size(); i++) {
        dpdt.noalias() += param_generators[i].transpose() * prob;
    }
    return(dpdt_);
}

np_array MEInference::augmented_backward(double time, np_array_c backward_in, np_array_c forward_in, np_array_c rates_in) {
    // parse input
    Eigen::Map<vec> backward((double*)backward_in.data(), forward_in.size());
    Eigen::Map<vec> forward((double*)forward_in.data(), forward_in.size());
    // set up output
	np_array res_out = np_array(backward_in.size());
	Eigen::Map<vec> dydt((double*)res_out.data(), forward_in.size()); 
    Eigen::Map<vec> rates_grad((double*)res_out.data()+forward_in.size(), rates_in.size());
    // copute output
    dydt.noalias() = -generator * backward;
    for (int i = 0; i < rates_in.size(); i++) {
        rates_grad[i] = forward.dot(param_generators[i] * backward);
    }
    return(res_out);
}