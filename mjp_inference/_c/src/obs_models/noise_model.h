#pragma once

#include "../../types.h"
#include "param.h"

// noise model base class

class NoiseModel {
    public:
    // constructor
    NoiseModel(std::vector<std::string>&& param_list_);
    NoiseModel(const std::vector<Param>& params);
    NoiseModel(std::vector<Param>&& params);

    // setup
    std::vector<vec> build_param_values();
    std::vector<std::string> build_param_list();

    // getters
    inline const std::vector<std::string>& get_param_list() {
        return(param_list);
    }

    // virtual function
    virtual vec sample(const std::vector<vec>& params, std::mt19937* rng) = 0;
    virtual double log_prob(const std::vector<vec>& params, const vec& obs) = 0;
    virtual std::vector<vec> log_prob_grad(const std::vector<vec>& params, const vec& obs) = 0;

    // main functions
    inline vec sample(std::mt19937* rng) {
        return(sample(param_values, rng));
    }
    inline double log_prob(const vec& obs) {
        return(log_prob(param_values, obs));
    }
    inline std::vector<vec> log_prob_grad(const vec& obs) {
        return(log_prob_grad(param_values, obs));
    }
    inline vec sample(unsigned seed) {
        std::mt19937 rng(seed);
        return(sample(&rng));
    }

    protected:
    std::vector<Param> param_map;
    std::vector<std::string> param_list;
    std::vector<vec> param_values;
};

// trampoline class

class PyNoiseModel : public NoiseModel {

    public:

    // inherit constructors
    using NoiseModel::NoiseModel;

    // main functions
    virtual vec sample(const std::vector<vec>& params, std::mt19937* rng) override;
    virtual double log_prob(const std::vector<vec>& params, const vec& obs) override;
    virtual std::vector<vec> log_prob_grad(const std::vector<vec>& params, const vec& obs) override;

};


// normal  model

class Normal : public NoiseModel {

    public:
    // constructors
    Normal() : NoiseModel(std::vector<std::string>({"mu", "sigma"})) {}
    Normal(const vec& mu, const vec& sigma);

    // helper functions
    std::vector<Param> build_param_map(const vec& mu, const vec& sigma);
    vec standard_normal_vec(unsigned dim, std::mt19937* rng);

    // main functions
    vec sample(const std::vector<vec>& params, std::mt19937* rng) override;
    double log_prob(const std::vector<vec>& params, const vec& obs) override;
    std::vector<vec> log_prob_grad(const std::vector<vec>& params, const vec& obs) override;
};