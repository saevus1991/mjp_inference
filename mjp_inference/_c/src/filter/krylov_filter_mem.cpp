#include "krylov_filter_mem.h"

// consructor

KrylovFilterMem::KrylovFilterMem(MEInference* master_equation_in, ObservationModel* obs_model_in, const vec& obs_times_in, const mat_rm& observations_in, const vec& initial_in, const vec& rates_in, const vec& obs_param_in) :
    KrylovFilter(master_equation_in, obs_model_in, obs_times_in, observations_in, initial_in, rates_in, obs_param_in),
    states_post(num_steps)
    {}

// main functions

double KrylovFilterMem::log_prob() {
  // preparations
  vec state(initial);
  double time = 0.0;
  // iteraet over observations
  for (int i = 0; i < num_steps; i++) {
    // propagate
    KrylovPropagator propagator(master_equation, state, rates, obs_times[i]-time);
    state.noalias() = propagator.propagate();
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
    states_post[i] = state;
    // update time
    time = obs_times[i];
  }
  return(norm.sum());
}

void KrylovFilterMem::log_prob_backward() {
  // preparations
  double time;
  vec backward;
  rates_grad.setZero();
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
    // recompute forward
    double delta_t;
    vec state;
    if (i >= 1) {
        delta_t = obs_times[i]-obs_times[i-1];
        state = states_post[i-1];
    } else {
        delta_t = obs_times[0];
        state = initial;
    }
    KrylovPropagator propagator(master_equation, state, rates, delta_t);
    propagator.propagate();
    // solve backward
    propagator.backward(backward);
    backward.noalias() = propagator.get_initial_grad();
    // compute rates gradient
    propagator.compute_rates_grad();
    rates_grad.noalias() += propagator.get_rates_grad();
  }
  initial_grad = backward;
}

void KrylovFilterMem::compute_rates_grad() {
  return;
}