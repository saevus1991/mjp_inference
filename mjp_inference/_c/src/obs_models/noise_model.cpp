#include "noise_model.h"

// constructor 

NoiseModel::NoiseModel(std::vector<std::string>&& param_list_) :
    param_list(param_list_)
    {}

NoiseModel::NoiseModel(const std::vector<Param>& params) :
    param_map(params),
    param_list(build_param_list())
    {}

NoiseModel::NoiseModel(std::vector<Param>&& params) :
    param_map(params),
    param_list(build_param_list()),
    param_values(build_param_values())
    {}

// main functions


// setup

std::vector<vec> NoiseModel::build_param_values() {
    std::vector<vec> param_values;
    for (unsigned i = 0; i < param_map.size(); i++) {
        param_values.push_back(param_map[i].get_value());
    }
    return(param_values);
}

std::vector<std::string> NoiseModel::build_param_list() {
    std::vector<std::string> param_list;
    for (unsigned i = 0; i < param_map.size(); i++) {
        param_list.push_back(param_map[i].get_name());
    }
    return(param_list);
}

// *** trampoline class **

vec PyNoiseModel::sample(const std::vector<vec>& params, std::mt19937* rng) {
    PYBIND11_OVERRIDE_PURE(
        vec, 
        NoiseModel,      
        sample,          
        rng
    ); 
}

double PyNoiseModel::log_prob(const std::vector<vec>& params, const vec& obs) {
    PYBIND11_OVERRIDE_PURE(
        double, 
        NoiseModel,      
        log_prob          
        params, obs
    ); 
}

std::vector<vec> PyNoiseModel::log_prob_grad(const std::vector<vec>& params, const vec& obs) {
    PYBIND11_OVERRIDE_PURE(
        std::vector<vec>, 
        NoiseModel,      
        log_prob_grad          
        params, obs
    ); 
}

// *** gauss obs ***

// constructor

Normal::Normal(const vec& mu, const vec& sigma) : NoiseModel(build_param_map(mu, sigma)) {}

// helper functions

std::vector<Param> Normal::build_param_map(const vec& mu, const vec& sigma) {
    std::vector<Param> param_map;
    if (mu.size() == sigma.size()) {
        param_map.push_back(Param("mu", mu));
        param_map.push_back(Param("sigma", sigma));
    } else if ( (mu.size() > 1) && (sigma.size()) == 1) {
        vec sigma_exp(mu.size());
        sigma_exp.setZero();
        sigma_exp = (sigma_exp.array() + sigma[0]).matrix();
        param_map.push_back(Param("mu", mu));
        param_map.push_back(Param("sigma", sigma_exp));
    } else if ( (mu.size() == 1) && (sigma.size() > 1) ) {      
        vec mu_exp(sigma.size());
        mu_exp.setZero();
        mu_exp = (mu_exp.array() + mu[0]).matrix();
        param_map.push_back(Param("mu", mu_exp));
        param_map.push_back(Param("sigma", sigma));
    } else {
        std::string msg = "Cannot handle loc of size " + std::to_string(mu.size()) + " and scale of size " + std::to_string(sigma.size());
        throw std::invalid_argument(msg);
    }
    return(param_map);
}

vec Normal::standard_normal_vec(unsigned dim, std::mt19937* rng) {
    std::normal_distribution<double> dist = std::normal_distribution<double>();
    vec sample(dim);
    double* sample_ptr = sample.data();
    for (unsigned i = 0; i < dim; i ++) {
        sample_ptr[i] = dist(*rng);
    }
    return(sample);
}

// main functions

vec Normal::sample(const std::vector<vec>& params, std::mt19937* rng) {
    // parse params
    const vec& mu = params[0];
    const vec& sigma = params[1];
    // create sample
    vec sample = standard_normal_vec(mu.size(), rng);
    vec obs = mu + (sigma.array() * sample.array()).matrix();
    return(obs);
}


double Normal::log_prob(const std::vector<vec>& params, const vec& obs) {
    // parse params
    const vec& mu = params[0];
    const vec& sigma = params[1];
    const double* mu_ptr = mu.data();
    const double* sigma_ptr = sigma.data();
    const double* obs_ptr = obs.data();
    // calculate llh
    double llh_ = -0.5 * mu.size() * std::log(2*M_PI);
    for (unsigned i = 0; i < mu.size(); i ++) {
        double tmp = (obs_ptr[i]-mu_ptr[i]) / sigma_ptr[i] ;
        llh_ += -0.5*tmp*tmp - std::log(sigma_ptr[i]);
    }
    return(llh_);
}

std::vector<vec> Normal::log_prob_grad(const std::vector<vec>& params, const vec& obs) {
    // parse params
    const vec& mu = params[0];
    const vec& sigma = params[1];
    const double* mu_ptr = mu.data();
    const double* sigma_ptr = sigma.data();
    const double* obs_ptr = obs.data();
    std::vector<vec> gradients(2);
    // compute mu gradient
    gradients[0] = vec(mu.size());
    double* grad_mu_ptr = gradients[0].data();
    for (unsigned i = 0; i < mu.size(); i++) {
        grad_mu_ptr[i] = -(mu_ptr[i] - obs_ptr[i]) / (sigma_ptr[i] * sigma_ptr[i]);
    }
    // compute sigma gradient
    gradients[1] = vec(sigma.size());
    double* grad_sigma_ptr = gradients[1].data();
    for (unsigned i = 0; i < sigma.size(); i++) {
        double tmp =  (obs_ptr[i]-mu_ptr[i]) / sigma_ptr[i];
        // grad_sigma_ptr[i] = (tmp*tmp - 1.0) / sigma_ptr[i];
        grad_sigma_ptr[i] = std::pow(obs_ptr[i]-mu_ptr[i], 2) / std::pow(sigma_ptr[i], 3) - 1.0 / sigma_ptr[i];
    }
    return(gradients);
}