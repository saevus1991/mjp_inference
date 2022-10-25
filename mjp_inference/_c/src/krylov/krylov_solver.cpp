#include "krylov_solver.h"

// constructor
KrylovSolver::KrylovSolver(MEInference* master_equation_in, const vec& initial_in, const vec& rates_in, const vec& obs_times_in) :
    num_steps(obs_times_in.size()),
    sub_steps(100),
    initial(initial_in),
    master_equation(master_equation_in),
    transition_model(master_equation->get_model()),
    obs_times(obs_times_in),
    rates(rates_in),
    initial_grad(transition_model->get_num_states()),
    rates_grad(transition_model->get_num_rates()),
    forward_fun(num_steps),
    backward_fun(num_steps)
{}

// KrylovSolver::KrylovSolver(MEInference* master_equation_in, const vec& initial_in, const vec& rates_in, const vec& obs_times_in) 
// {
//   std::cout << "KrylovSolver constructor" << std::endl;
//   num_steps = obs_times_in.size();
//   sub_steps = 100;
//   initial = initial_in;
//   master_equation master_equation),
//     transition_model(master_equation->get_model()),
//     obs_times(obs_times_in),
//     rates(rates_in),
//     initial_grad(transition_model->get_num_states()),
//     rates_grad(transition_model->get_num_rates()),
//     forward_fun(num_steps),
//     backward_fun(num_steps)
//  }

KrylovSolver::KrylovSolver(MEInference* master_equation_in, const vec& initial_in, const vec& rates_in, double obs_times_in) : KrylovSolver(master_equation_in, initial_in, rates_in, ut::double2vec(obs_times_in))
{ }

// main functions

np_array KrylovSolver::forward(int krylov_order) { 
    /* 
    Propagate the system forward for a given time interval
    */
    // bind forward function
    generator = master_equation->get_generator(rates);
    Operator fun = [this] (vec& x) { return(generator.transpose() * x); };
    // set up output
    std::vector<int> shape({int(obs_times.rows()), int(initial.rows())});
    np_array res_out(shape);
    Eigen::Map<mat_rm> res((double*)res_out.data(), obs_times.rows(), initial.rows());
    // create krylov operator and propagate
    double time = 0.0;
    vec state(initial);
    for (int i = 0; i < num_steps; i++) {
        forward_fun[i] = Krylov(fun, state, krylov_order);
        state.noalias() = forward_fun[i].eval(obs_times[i]-time);
        res.row(i) = state.transpose();
        time = obs_times[i];
    }
    return(res_out);
}

void KrylovSolver::backward(int krylov_order, np_array grad_output_in) {
    // parse input
    Eigen::Map<mat_rm> grad_output((double*)grad_output_in.data(), obs_times.rows(), initial.rows());
    // preparations
    vec backward(initial.rows());
    backward.setZero();
    // Operator fun = [this] (vec& x) { return(transition_model.backward_neg(x)); };
    Operator fun = [this] (vec& x) { return(generator * x); };
    // iterate backward in time
    for (int i = num_steps-1; i >= 1; i--) {
        // obs update and backward solution
        backward.noalias() += grad_output.row(i).transpose();
        backward_fun[i] = Krylov(fun, backward, krylov_order);
        backward.noalias() = backward_fun[i].eval(obs_times[i] - obs_times[i-1]);
    }
    // perform last step
    backward.noalias() += grad_output.row(0).transpose();
    backward_fun[0] = Krylov(fun, backward, krylov_order);
    initial_grad.noalias() = backward_fun[0].eval(obs_times[0]);
}

void KrylovSolver::compute_rates_grad() {
  // preparations
  int num_rates = transition_model->get_num_rates();
  rates_grad.setZero();
  double start_time = 0.0;
  double end_time;
  double time;
  const std::vector<csr_mat>& param_generators = master_equation->get_param_generators();
  // iterate
  for (int i = 0; i < num_steps; i++) {
    // compute reduced generators
    if (backward_fun[i].is_degenerate()) {
      start_time = obs_times[i];
      continue;
    } 
    std::vector<mat> reduced_generators(param_generators.size());
    mat forward_span = forward_fun[i].get_span();
    mat backward_span = backward_fun[i].get_span();
    for (int k = 0; k < reduced_generators.size(); k++) {
      reduced_generators[k] = backward_span.transpose() * (param_generators[k].transpose() * forward_span);
    }
    // set up time variables
    end_time = obs_times[i];
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