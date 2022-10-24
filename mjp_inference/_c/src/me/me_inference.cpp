#include "me_inference.h"

// constructor

MEInference::MEInference(MJP* mjp_) : 
    MasterEquation(mjp_)
    {
        build_param_generators();
    }


// helpers

void MEInference::build_param_generators() {
    // set up genrators
    unsigned num_rates = mjp->get_num_rates();
    for (unsigned i = 0; i < num_rates; i++) {
        param_generators.push_back(csr_mat(mjp->get_num_states(), mjp->get_num_states()));
    }
    // add contributions
    unsigned num_events = mjp->get_num_events();
    for (unsigned i = 0; i < num_events; i++) {
        unsigned rate_index = mjp->event2rate(i);
        param_generators[rate_index] = param_generators[rate_index] + base_generators[i];
    }
    return;
}