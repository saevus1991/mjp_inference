#pragma once
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

#include "../../types.h"
// #include "../util/util.h"


class Krylov {

    public:

    // generator
    Krylov();
    Krylov(csr_mat generator_in, vec initial_in, int order_in = 5);
    Krylov(Operator generator_in, vec initial_in, int order_in = 5);

    // setup
    void build();
    void expand(int inc);

    // getters
    inline mat get_span() {
        return(span);
    }
    inline mat get_proj() {
        return(proj);
    }

    // main functions
    vec eval(double time);
    vec eval_proj(double time);
    vec eval_sub(double time);
    double eval_err(double time);
    mat project(csr_mat& matrix);
    inline bool is_degenerate() {return(degenerate);}

    public:
    double tol;
    int dim;
    csr_mat generator_mat;
    Operator generator;
    double scale;
    double norm;
    vec q;
    int order;
    mat span;
    mat proj;
    bool degenerate;
    bool finite;

};