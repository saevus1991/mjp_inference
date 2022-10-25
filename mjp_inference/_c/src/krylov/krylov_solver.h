#pragma once
#include "../../types.h"
#include "../models/mjp.h"
#include "../me/me_inference.h"
#include "../krylov/krylov_exp.h"


class KrylovSolver {

    public:

    KrylovSolver(MEInference* master_equation_in, const vec& initial_in, const vec& rates_in, const vec& obs_times_in);
    KrylovSolver(MEInference* master_equation_in, const vec& initial_in, const vec& rates_in, double obs_times_in);

    inline int debug(MEInference* master_equation_in, const vec& initial_in, const vec& rates_in, const vec& obs_times_in) {
        return(obs_times_in.size());
    }

    // main functions 
    np_array forward(int krylov_order);
    void backward(int krylov_order, np_array grad_output_in);
    void compute_rates_grad();

    // // getters
    inline vec get_initial_grad() {
        return(initial_grad);
    }
    inline vec get_rates_grad() {
        return(rates_grad);
    }


    private:
    int num_steps;
    int sub_steps;
    MEInference* master_equation;
    MJP* transition_model;
    vec initial;
    vec rates;
    csr_mat generator;
    vec obs_times;
    vec initial_grad;
    vec rates_grad;
    Krylov forward_fun_tmp;
    std::vector<Krylov> forward_fun;
    std::vector<Krylov> backward_fun;
};

