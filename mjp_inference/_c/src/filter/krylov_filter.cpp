#include "krylov_filter.h"


// constructor
KrylovFilter::KrylovFilter(MEInference* master_equation_in, ObservationModel* obs_model_in, const vec& obs_times_in, const mat_rm& observations_in, const vec& initial_in, const vec& rates_in, const vec& obs_param_in) :
  num_steps(obs_times_in.size()),
  // sub_steps(100),
  master_equation(master_equation_in),
  transition_model(master_equation->get_model()),
  obs_model(obs_model_in),
  obs_times(obs_times_in),
  observations(observations_in),
  initial(initial_in),
  rates(rates_in),
  obs_param(obs_param_in),
  states(num_steps),
  indices(num_steps),
  llh_stored(num_steps),
  norm(num_steps),
  rates_grad(rates_in.rows())
{ }


// main functions


double KrylovFilter::log_prob() {
  // preparations
  vec state(initial);
  double time = 0.0;
  // iteraet over observations
  for (int i = 0; i < num_steps; i++) {
    // propagate
    propagators.push_back(KrylovPropagator(master_equation, state, rates, obs_times[i]-time));
    state.noalias() = propagators[i].propagate();
    indices[i] = ut::math::nn_project(state);
    states[i] = state;
    // compute obs update
    vec obs = observations.row(i).transpose();
    llh_stored[i] = obs_model->log_prob_vec(obs_times[i], obs_param, obs);
    vec tmp = state.array().log() + llh_stored[i].array();
    double max_state = tmp.maxCoeff();
    state.noalias() = (tmp.array()-max_state).exp().matrix();
    double norm_tmp = state.sum();
    state = state / norm_tmp;
    norm[i] = max_state + std::log(norm_tmp);
    // update time
    time = obs_times[i];
  }
  return(norm.sum());
}

void KrylovFilter::log_prob_backward() {
  // preparations
  double time;
  vec backward;
  obs_param_grad = vec(obs_model->get_num_param());
  obs_param_grad.setZero();
  for (int i = num_steps-1; i >= 0; i--) {
    // observation update
    if (i == num_steps-1) {
      vec d_llh = (states[i].array().log() + llh_stored[i].array() - norm[i]).exp();
      backward = (llh_stored[i].array() - norm[i]).exp();
      vec obs = observations.row(i).transpose();
      obs_param_grad += obs_model->log_prob_grad_vec(obs_times[i], obs_param, obs).transpose() * d_llh;
    } else {
      double tmp = ( backward.array() * (states[i].array().log() + llh_stored[i].array() - norm[i]).exp()).sum();
      vec d_llh = (states[i].array().log() + llh_stored[i].array() - norm[i]).exp() * (backward.array() -  tmp + 1.0);
      backward = (llh_stored[i].array() - norm[i]).exp() * (backward.array() - tmp + 1.0);
      vec obs = observations.row(i).transpose();
      obs_param_grad += obs_model->log_prob_grad_vec(obs_times[i], obs_param, obs).transpose() * d_llh;
    }
    // thresholding 
    backward(indices[i]).setZero();
    // solve backward
    propagators[i].backward(backward);
    backward.noalias() = propagators[i].get_initial_grad();
  }
  initial_grad = backward;
}

void KrylovFilter::compute_rates_grad() {
  // preparations
  rates_grad.setZero();
  // iterate over obs intervals
  for (int i = 0; i < num_steps; i++) {
    // compute rates gradient of local propagator
    propagators[i].compute_rates_grad();
    // accumulate
    rates_grad.noalias() += propagators[i].get_rates_grad();
  }
  return;
}
