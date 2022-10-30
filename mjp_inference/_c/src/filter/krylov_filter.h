#pragma once
#include "../../types.h"
#include "../util/math.h"
#include "../models/mjp.h"
#include "../obs_models/obs_model.h"
#include "../krylov/krylov_propagator.h"
#include "../me/me_inference.h"

/* To do:
- template obs model
*/

class KrylovFilter {

    public:
    KrylovFilter(MEInference* master_equation_in, ObservationModel* obs_model_in, const vec& obs_times_in, const mat_rm& observations_in, const vec& initial_in, const vec& rates_in, const vec& obs_param_in);

    // main functions 
    double log_prob();
    void log_prob_backward();
    void compute_rates_grad();

    // // getters
    inline vec get_initial_grad() {
        return(initial_grad);
    }
    inline vec get_rates_grad() {
        return(rates_grad);
    }
    inline vec get_obs_param_grad() {
        return(obs_param_grad);
    }

    private:
    int num_steps;
    MEInference* master_equation;
    MJP* transition_model;
    ObservationModel* obs_model;
    vec initial;
    vec rates;
    vec obs_param;
    vec obs_times;
    mat_rm observations;
    std::vector<vec> states;
    std::vector<std::vector<unsigned>> indices;
    std::vector<vec> llh_stored;
    vec norm;
    vec initial_grad;
    vec rates_grad;
    vec obs_param_grad;
    std::vector<KrylovPropagator> propagators;
};

