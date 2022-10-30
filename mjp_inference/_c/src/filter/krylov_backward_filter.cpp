#include "krylov_backward_filter.h"


// constructor
KrylovBackwardFilter::KrylovBackwardFilter(MEInference* master_equation_in, ObservationModel* obs_model_in, const vec& obs_times_in, const mat_rm& observations_in, const vec& initial_in, const vec& rates_in, const vec& obs_param_in, const vec& tspan_in) :
  num_steps(obs_times_in.size()),
  master_equation(master_equation_in),
  transition_model(master_equation->get_model()),
  obs_model(obs_model_in),
  obs_times(obs_times_in),
  observations(observations_in),
  initial(initial_in),
  rates(rates_in),
  obs_param(obs_param_in),
  tspan(tspan_in),
  states(num_steps),
  indices(num_steps),
  llh_stored(num_steps),
  norm(num_steps)
{ }

KrylovBackwardFilter::KrylovBackwardFilter(MEInference* master_equation_in, ObservationModel* obs_model_in, const vec& obs_times_in, const mat_rm& observations_in, const vec& initial_in, const vec& rates_in, const vec& obs_param_in) : KrylovBackwardFilter::KrylovBackwardFilter( master_equation_in, obs_model_in, obs_times_in, observations_in, initial_in, rates_in, obs_param_in, obs_times_in(std::vector<int> {0, int(obs_times_in.rows())-1}))
{}

// main functions


double KrylovBackwardFilter::log_prob() {
  return(norm.sum());
}

void KrylovBackwardFilter::forward_filter() {
  // preparations
  vec state(initial);
  double time = tspan[0];
  // iteraet over observations
  for (int i = 0; i < num_steps; i++) {
    // propagate
    forward_propagators[i] = KrylovPropagator(master_equation, state, rates, obs_times[i]-time);
    state.noalias() = forward_propagators[i].propagate();
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
  // solve for final interval
  forward_propagators[num_steps] = KrylovPropagator(master_equation, state, rates, tspan[1]-time);
  state.noalias() = forward_propagators[num_steps].propagate();
  return;
}

std::tuple<int, double, double> KrylovBackwardFilter::find_interval(double time) {
  int index;
  double t_lower;
  double t_upper;
  // find time and propagator index
  if (time > tspan[1] || time < tspan[0]) {
    std::string msg = "Time " + std::to_string(time) + " outside of tspan interval [" + std::to_string(tspan[0]) + ", " + std::to_string(tspan[1]) + "]";
    throw std::invalid_argument(msg);
  } else if (time <= obs_times[0]) {
    index = 0;
    t_lower = tspan[0];
    t_upper = obs_times[0];
  } else if (time >= obs_times[num_steps-1]) {
    index = num_steps;
    t_lower = obs_times[num_steps-1];
    t_upper = tspan[1];
  } else {
    auto idx = std::lower_bound(obs_times.begin(), obs_times.end(), time);
    index = idx - obs_times.begin();
    t_lower = obs_times[index-1];
    t_upper = obs_times[index];
  }
  return(std::make_tuple(index, t_lower, t_upper));
}

vec KrylovBackwardFilter::eval_forward_filter(double time) {
  // find time interval
  std::tuple<int, double, double> interval = find_interval(time);
  int index = std::get<0>(interval);
  double t_eff = time - std::get<1>(interval);
  // evaluate result
  vec res = forward_propagators[index].eval(t_eff);
  return(res);
}

template <class T>
mat_rm KrylovBackwardFilter::eval_forward_filter(T& time) {
  // set up output
  mat_rm filt_grid(time.rows(), transition_model->get_num_states());
  // iterate over times
  for (int i = 0; i < time.rows(); i++) {
    filt_grid.row(i) = eval_forward_filter(time[i]).transpose();
  }
  return(filt_grid);
}

template mat_rm KrylovBackwardFilter::eval_forward_filter<vec>(vec& time);
template mat_rm KrylovBackwardFilter::eval_forward_filter<Eigen::Map<vec>>(Eigen::Map<vec>& time);

void KrylovBackwardFilter::backward_filter() {
  // preparations
  double time = tspan[1];
  double delta_t;
  vec backward(transition_model->get_num_states());
  backward.setConstant(1.0);
  // iterate over observations
  for (int i = num_steps-1; i >= 0; i--) {
    // propagate
    delta_t = time - obs_times[i];
    backward_propagators[i+1] = KrylovPropagator(master_equation, backward, rates, delta_t);
    backward.noalias() = backward_propagators[i+1].reverse();
    ut::math::project_positive(backward);
    // observation update
    vec obs = observations.row(i).transpose();
    vec llh = obs_model->log_prob_vec(obs_times[i], obs_param, obs);
    vec tmp = backward.array().log() + llh.array();
    double max_state = tmp.maxCoeff();
    backward.noalias() = (tmp.array()-max_state).exp().matrix();
    // update time
    time = obs_times[i];
  }
  // compute for first interval
  delta_t = time - tspan[0];
  backward_propagators[0] = KrylovPropagator(master_equation, backward, rates, delta_t);
  backward.noalias() = backward_propagators[0].reverse();
  ut::math::project_positive(backward);
}

vec KrylovBackwardFilter::eval_backward_filter(double time) {
  // find interval
  std::tuple<int, double, double> interval = find_interval(time);
  int index = std::get<0>(interval);
  double t_eff = std::get<2>(interval) - time;
  // evaluate result
  vec res = backward_propagators[index].eval(t_eff);
  return(res);
}

template <class T>
mat_rm KrylovBackwardFilter::eval_backward_filter(T& time) {
  // set up output
  mat_rm filt_grid(time.rows(), transition_model->get_num_states());
  // iterate over times
  for (int i = 0; i < time.rows(); i++) {
    filt_grid.row(i) = eval_backward_filter(time[i]).transpose();
  }
  return(filt_grid);
}

template mat_rm KrylovBackwardFilter::eval_backward_filter<vec>(vec& time);
template mat_rm KrylovBackwardFilter::eval_backward_filter<Eigen::Map<vec>>(Eigen::Map<vec>& time);

vec KrylovBackwardFilter::eval_smoothed(double time) {
  // find interval
  std::tuple<int, double, double> interval = find_interval(time);
  int index = std::get<0>(interval);
  double t_forward = time - std::get<1>(interval);
  double t_backward = std::get<2>(interval) - time;
  // evaluate result
  vec forward = forward_propagators[index].eval(t_forward);
  ut::math::project_positive(forward);
  vec backward = backward_propagators[index].eval(t_backward);
  ut::math::project_positive(backward);
  vec smoothed = (forward.array().log() + backward.array().log()).matrix();
  double max_log = smoothed.maxCoeff();
  forward.noalias() = (smoothed.array() - max_log).exp().matrix();
  double norm = forward.sum();
  smoothed.noalias() = forward / norm;
  return(smoothed);
}

template <class T>
mat_rm KrylovBackwardFilter::eval_smoothed(T& time) {
  // set up output
  mat_rm filt_grid(time.rows(), transition_model->get_num_states());
  // iterate over times
  for (int i = 0; i < time.rows(); i++) {
    filt_grid.row(i) = eval_smoothed(time[i]).transpose();
  }
  return(filt_grid);
}

template mat_rm KrylovBackwardFilter::eval_smoothed<vec>(vec& time);
template mat_rm KrylovBackwardFilter::eval_smoothed<Eigen::Map<vec>>(Eigen::Map<vec>& time);

vec KrylovBackwardFilter::get_smoothed_initial() {
  // evaluate backward
  int index = 0;
  double t_backward = obs_times[0] - tspan[0];
  vec backward = backward_propagators[index].eval(t_backward);
  ut::math::project_positive(backward);
  ut::math::project_positive(initial);
  // multiply and normalize
  vec smoothed = (initial.array().log() + backward.array().log()).matrix();
  double max_log = smoothed.maxCoeff();
  backward.noalias() = (smoothed.array() - max_log).exp().matrix();
  double norm = backward.sum();
  smoothed.noalias() = backward / norm;
  return(smoothed);
}