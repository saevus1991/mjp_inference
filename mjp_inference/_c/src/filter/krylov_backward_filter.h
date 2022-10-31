#pragma once
#include "../../types.h"
#include "../util/math.h"
#include "../models/mjp.h"
#include "../obs_models/obs_model.h"
#include "../krylov/krylov_propagator.h"
#include "../me/me_inference.h"


class KrylovBackwardFilter {

    public:
    KrylovBackwardFilter(MEInference* master_equation_in, ObservationModel* obs_model_in, vec obs_times_in, mat_rm observations_in, vec initial_in, vec rates_in, vec obs_param_in);
    KrylovBackwardFilter(MEInference* master_equation_in, ObservationModel* obs_model_in, vec obs_times_in, mat_rm observations_in, vec initial_in, vec rates_in, vec obs_param_in, vec tspan);

    // main functions 
    double log_prob();
    std::tuple<int, double, double> find_interval(double time);
    void forward_filter();
    vec eval_forward_filter(double time); // #FIXME: does not work for t=0.0, possibly also t=tspan[0]
    template <class T>
    mat_rm eval_forward_filter(T& time);
    void backward_filter();
    vec eval_backward_filter(double time);
    template <class T>
    mat_rm eval_backward_filter(T& time);
    vec eval_smoothed(double time);
    template <class T>
    mat_rm eval_smoothed(T& time);
    vec get_smoothed_initial();
    // #TODO: add eval functions that allow array input

    private:
    int num_steps;
    // int sub_steps;
    MEInference* master_equation;
    MJP* transition_model;
    ObservationModel* obs_model;
    vec initial;
    vec rates;
    vec obs_param;
    csr_mat generator;
    vec obs_times;
    mat_rm observations;
    std::vector<vec> states;
    std::vector<std::vector<unsigned>> indices;
    std::vector<vec> llh_stored;
    vec norm;
    std::map<int, KrylovPropagator> forward_propagators;
    std::map<int, KrylovPropagator> backward_propagators;
    vec tspan;
};

