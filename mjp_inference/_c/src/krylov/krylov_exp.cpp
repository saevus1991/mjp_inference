#include "krylov_exp.h"

// constructor

Krylov::Krylov() {}  

Krylov::Krylov(csr_mat generator_in, vec initial_in, int order_in) :
    tol(1e-10),
    dim(initial_in.rows()),
    generator_mat(generator_in),
    generator(Operator([this](vec& x) { return generator_mat*x ;})),
    order(order_in)
    {
        double check = initial_in.norm();
        if (check < 1e-16) {
            span = vec(dim);
            span.setZero();
            degenerate = true;
        } else {
            scale = initial_in.array().abs().maxCoeff();
            q = initial_in / scale;
            norm = q.norm();
            q /= norm;
            build();
            degenerate = false;
        }
    } 

Krylov::Krylov(Operator generator_in, vec initial_in, int order_in) :
    tol(1e-10),
    dim(initial_in.rows()),
    generator(generator_in),
    order(order_in)
    {
        double check = initial_in.norm();
        if (check < 1e-16) {
            span = vec(dim);
            span.setZero();
            degenerate = true;
        } else {
            scale = initial_in.array().abs().maxCoeff();
            q = initial_in / scale;
            norm = q.norm();
            q /= norm;
            build();
            degenerate = false;
        }
    }

// setup

void Krylov::build() {
    // preparations
    span = mat(dim, order);
    proj = mat(order+1, order);
    proj.setZero();
    vec v(dim);
    // iterate over order
    for (int i = 0; i < order; i++) {
        span.col(i) = q;
        v.noalias() = generator(q);
        for (int j = 0; j < i+1; j++) {
            proj(j, i) = v.dot(span.col(j));
            q.noalias() = v - proj(j, i) * span.col(j);
            v.noalias() = q;
        }
        proj(i+1, i) = v.norm();
        q.noalias() = v / proj(i+1, i);
    }
    return;
}

void Krylov::expand(int inc){
    // preparations
    int new_order = order + inc;
    mat new_span = mat(dim, new_order);
    mat new_proj = mat(new_order+1, new_order);
    new_proj.setZero();
    vec v(dim);
    // reuse existing 
    new_span.block(0, 0, dim, order) = span;
    new_proj.block(0, 0, order+1, order) = proj;
    // iterate over order
    for (int i = order; i < new_order; i++) {
        new_span.col(i) = q;
        v.noalias() = generator(q);
        for (int j = 0; j < i+1; j++) {
            new_proj(j, i) = v.dot(new_span.col(j));
            q.noalias() = v - new_proj(j, i) * new_span.col(j);
            v.noalias() = q;
        }
        new_proj(i+1, i) = v.norm();
        q.noalias() = v / new_proj(i+1, i);
    }
    // update stored
    order = new_order;
    span = new_span;
    proj = new_proj;
    return;
}

vec Krylov::eval(double time) {
    /*
    Evaluate the propagator after time. Uses a simple adaptve scheme to increase the order if not sufficient. 
    */
    // return zero if output is degenerate
    if (degenerate) {
        vec output(dim);
        output.setZero();
        return(output);
    }
    bool converged = false;
    vec res;
    while (!converged) {
        // compute projected result and error
        res = eval_proj(time);
        double err = res[order];
        if (std::abs(err) <= tol) {
            converged = true;
        } else{
            expand(5);
        }
    }
    vec output = scale * span * res.segment(0, order);
    return(output);
}

vec Krylov::eval_proj(double time) {
    /*
    Evaluate the propagator after time. No error checking is performed. The result is returned in projected space.
    */
    // create initial
    vec initial(order+1);
    initial.setZero();
    initial[0] = norm;
    // create extended projected generator
    mat proj_gen(order+1, order+1);
    proj_gen.setZero();
    proj_gen.block(0, 0, order+1, order) = proj * time;
    proj_gen(order, order) = 1.0;
    // compute result in krylov space
    mat exp = proj_gen.exp();
    vec res = exp * initial;
    return(res);
}

vec Krylov::eval_sub(double time) {
    /*
    Evaluate the propagator after time. No error checking is performed. The result is returned in projected space.
    */
    return(scale * eval_proj(time).segment(0, order));
}

double Krylov::eval_err(double time) {
    /*
    Evaluate the propagator after time. No error checking is performed. The result is returned in projected space.
    */
    // create initial
    return(eval_proj(time)[order]);
}

mat Krylov::project(csr_mat& matrix) {
    mat projected = span.transpose() * (matrix * span);
    return(projected);
}