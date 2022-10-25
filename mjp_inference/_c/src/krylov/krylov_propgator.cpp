#include "krylov_propagator.h"

// constructor

KrylovPropagator::KrylovPropagator() {
  transition_model = nullptr;
}


KrylovPropagator::KrylovPropagator(MEInference* master_equation_in, const vec& initial_in, const vec& rates_in, double time_in) :
    sub_steps(100),
    krylov_order(20),
    initial(initial_in),
    master_equation(master_equation_in),
    transition_model(master_equation->get_model()),
    time(time_in),
    rates(rates_in),
    initial_grad(transition_model->get_num_states()),
    rates_grad(transition_model->get_num_rates())
{ }


vec KrylovPropagator::propagate() {
    /* 
    Propagate the system forward for a given time interval
    */
    // bind forward function
    generator = master_equation->get_generator(rates); // #TODO: check if every propagator object needs copy of generator
    Operator fun = [this] (vec& x) { return(generator.transpose() * x); };
    // create krylov operator and propagate
    forward_fun = std::vector<Krylov>();
    eval_times = std::vector<double>();
    double tol = 1e-10;  // # TODO: make tol an argument
    double t_tmp = time;
    double t_eval = 0.0;
    state = initial;
    // adaptivly create evaluation times
    while (t_eval < time) {
        // find time step with sufficiently small error
        Krylov krylov(fun, state, krylov_order);
        double err = krylov.eval_err(t_tmp);
        while (err > tol) {
            t_tmp = 0.5*t_tmp;
            err = krylov.eval_err(t_tmp);
        }
        // update state and time
        state.noalias() = krylov.eval(t_tmp);
        t_eval += t_tmp;
        t_tmp = std::min(3*t_tmp, time-t_eval);
        // store stuff
        forward_fun.push_back(krylov);
        eval_times.push_back(t_eval);
    }
    // set some variables
    num_steps = eval_times.size();
    backward_fun = std::vector<Krylov>(num_steps);
    return(state);
}


vec KrylovPropagator::reverse() {
    // preparations
    // bind forward function
    generator = master_equation->get_generator(rates);
    Operator fun = [this] (vec& x) { return(generator * x); };
    // create krylov operator and propagate
    double tol = 1e-10;
    double t_tmp = time;
    double t_eval = 0.0;
    state = initial;
    // adaptivly create evaluation times
    while (t_eval < time) {
        // find time step with sufficiently small error
        Krylov krylov(fun, state, krylov_order);
        double err = krylov.eval_err(t_tmp);
        while (err > tol) {
            t_tmp = 0.5*t_tmp;
            err = krylov.eval_err(t_tmp);
        }
        // update state and time
        state.noalias() = krylov.eval(t_tmp);
        t_eval += t_tmp;
        t_tmp = std::min(3*t_tmp, time-t_eval);
        // store stuff
        forward_fun.push_back(krylov);
        eval_times.push_back(t_eval);
    }
    // set some variables
    num_steps = eval_times.size();
    backward_fun = std::vector<Krylov>(num_steps);
    return(state);
}

vec KrylovPropagator::eval(double time) {
    // find time 
    auto idx = std::lower_bound(eval_times.begin(), eval_times.end(), time);
    int index = idx - eval_times.begin();
    // evaluate at time
    double t_eff = time;
    if (index > 0) {
        t_eff = time - eval_times[index-1];
    }
    vec res = forward_fun[index].eval(t_eff);
    return(res);
}

void KrylovPropagator::backward(np_array grad_output_in) {
    // parse input
    Eigen::Map<vec> grad_output((double*)grad_output_in.data(), initial.rows());
    // run vector verstion
    backward(grad_output);
}

template <class T>
void KrylovPropagator::backward(T& grad_output) {
    // preparations
    vec backward(grad_output);
    Operator fun = [this] (vec& x) { return(generator * x); };
    // iterate backward in time
    for (int i = num_steps-1; i >= 0; i--) {
        backward_fun[i] = Krylov(fun, backward);
        double t_eff = (i == 0) ? eval_times[0] : (eval_times[i] - eval_times[i-1]);
        backward.noalias() = backward_fun[i].eval(t_eff);
        if (backward.array().isNaN().any()) {
          backward.setZero();
          backward_fun[i] = Krylov(fun, backward);
          std::cout << "Warning: reset backward" << std::endl;
        }
    }
    // set gradients
    initial_grad.noalias() = backward;
    time_grad = grad_output.dot(generator.transpose() * state);
}

template void KrylovPropagator::backward<vec>(vec& grad_output);
template void KrylovPropagator::backward<Eigen::Map<vec>>(Eigen::Map<vec>& grad_output);

void KrylovPropagator::compute_rates_grad() {
  // preparations
  int num_rates = transition_model->get_num_rates();
  rates_grad.setZero();
  double start_time = 0.0;
  double end_time;
  double time;
  std::vector<csr_mat> param_generators = master_equation->get_param_generators();
  // iterate
  for (int i = 0; i < num_steps; i++) {
    // compute reduced generators
    if (backward_fun[i].is_degenerate()) {
      start_time = eval_times[i];
      continue;
    } 
    std::vector<mat> reduced_generators(param_generators.size());
    mat forward_span = forward_fun[i].get_span();
    mat backward_span = backward_fun[i].get_span();
    for (int k = 0; k < reduced_generators.size(); k++) {
      reduced_generators[k] = backward_span.transpose() * (param_generators[k].transpose() * forward_span);
    }
    // set up time variables
    end_time = eval_times[i];
    double delta = (end_time-start_time) / sub_steps;
    // compute lower boundary contribution
    vec forward = forward_fun[i].eval_sub(0.0);
    vec backward = backward_fun[i].eval_sub(end_time-start_time);
    for (int k = 0; k < num_rates; k++) {
      rates_grad[k] += 0.5 * delta * backward.dot(reduced_generators[k] * forward);
    }
    // compute inner contribution
    for (int j = 1; j < sub_steps; j++) {
      time = j*delta;
      forward = forward_fun[i].eval_sub(time);
      backward = backward_fun[i].eval_sub(end_time-start_time-time);
      for (int k = 0; k < num_rates; k++) {
        rates_grad[k] += delta * backward.dot(reduced_generators[k] * forward);
      }
    }
    // compute upper boundary contribution
    forward = forward_fun[i].eval_sub(end_time-start_time);
    backward = backward_fun[i].eval_sub(0.0);
    for (int k = 0; k < num_rates; k++) {
      rates_grad[k] += 0.5 * delta * backward.dot(reduced_generators[k] * forward);
    }
    start_time = end_time;
  }
  return;
}