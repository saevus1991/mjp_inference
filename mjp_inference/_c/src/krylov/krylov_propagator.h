#pragma once
#include <algorithm>
#include "../../types.h"
#include "../models/mjp.h"
#include "../me/me_inference.h"
// #include "../util/util.h"
// #include "../models/transition_model.h"
#include "../krylov/krylov_exp.h"


class KrylovPropagator {

    public:
    KrylovPropagator();
    KrylovPropagator(MEInference* master_equation_in, const vec& initial_in, const vec& rates_in, double time_in);

    // main functions 
    vec propagate();
    vec reverse();
    vec eval(double time);
    void backward(np_array grad_output_in);
    template <class T>
    void backward(T &grad_output);
    void compute_rates_grad();

    // getters
    inline double get_time_grad() {
        return(time_grad);
    }
    inline const vec& get_initial_grad() {
        return(initial_grad);
    }
    inline const vec& get_rates_grad() {
        return(rates_grad);
    }

    private:
    int num_steps;
    int sub_steps;
    int krylov_order;
    MEInference* master_equation;
    MJP* transition_model;
    vec initial;
    vec rates;
    vec state;
    csr_mat generator;
    double time;
    std::vector<double> eval_times;
    double time_grad;
    vec initial_grad;
    vec rates_grad;
    std::vector<Krylov> forward_fun;
    std::vector<Krylov> backward_fun;
};

