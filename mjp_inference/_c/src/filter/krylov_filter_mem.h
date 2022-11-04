#pragma once
#include "krylov_filter.h"


class KrylovFilterMem : public KrylovFilter {

    public:
    // constructor
    KrylovFilterMem(MEInference* master_equation_in, ObservationModel* obs_model_in, const vec& obs_times_in, const mat_rm& observations_in, const vec& initial_in, const vec& rates_in, const vec& obs_param_in);

    // main functions 
    virtual double log_prob() override;
    virtual void log_prob_backward() override;
    virtual void compute_rates_grad() override;

    protected:
    std::vector<vec> states_post;

};