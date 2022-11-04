#pragma once

#include "../../types.h"
#include "../models/mjp.h"
#include "noise_model.h"
#include "transform.h"
#include "../util/conversion.h"


class ObservationModel {

    public:
    // constructor
    ObservationModel(MJP* transition_model_, const std::string& noise_type_);
    ~ObservationModel();

    // setup helpers
    NoiseModel* build_noise_model();
    void build();

    // helper functions
    inline std::vector<vec> compute_noise_params(double time, const vec& state, const vec& param) const {
        std::vector<vec> noise_params(transform_map.size());
        for (unsigned i = 0; i < transform_map.size(); i++) {
            noise_params[i] = transform_map[i].transform(time, state, param);
        }
        return(noise_params);
    }

    // interface
    void add_param(Param param) {
        param_map.push_back(param);
        param_list.push_back(param.get_name());
        num_param++;
    }
    void make_add_param(const std::string& name, const vec& value) {
        add_param(Param(name, value));
    }
    void make_add_param(const std::string& name, double value) {
        add_param(Param(name, value));
    }
    inline void add_transform(Transform transform) {
        transform_map.push_back(transform);
    }
    inline unsigned transform_index(const std::string& transform) const {
        auto it = std::find(noise_param_list.begin(), noise_param_list.end(), transform);
        unsigned index = it - noise_param_list.begin();
        if (index == noise_param_list.size()) {
            std::string msg = "Transform \"" + transform + "\" not part of obs_model \"" ;
            throw std::invalid_argument(msg);
        }
        return(index);
    }

    // getters
    inline const std::string& get_noise_type() const {
        return(noise_type);
    }
    inline const std::vector<std::string>& get_noise_param_list() const {
        return(noise_param_list);
    }
    inline const std::vector<std::string>& get_param_list() const {
        return(param_list);
    }
    inline const vec& get_param_array() const {
        return(param_array);
    }
    inline np_array get_param_array_np() const {
        return(ut::vec2array(param_array));
    }
    inline const std::vector<Param>& get_param_map() const {
        return(param_map);
    }
    inline unsigned get_num_param() const {
        return(num_param);
    }
    inline unsigned get_obs_dim() const {
        return(obs_dim);
    }
    std::string get_param_parser() const;

    // main functions
    inline double log_prob(double time, const vec& state, const vec& param, const vec& obs) {
        // compute noise params
        std::vector<vec> noise_params = compute_noise_params(time, state, param);
        // evaluate llh
        double llh = noise_model->log_prob(noise_params, obs);
        return(llh);
    }
    inline vec log_prob_grad(double time, const vec& state, const vec& param, const vec& obs) {
        // gradients of the noise model
        std::vector<vec> noise_params = compute_noise_params(time, state, param);
        std::vector<vec> noise_grad = noise_model->log_prob_grad(noise_params, obs);
        // backprop through transform
        vec grad(param.size());
        grad.setZero();
        for (unsigned i = 0; i < transform_map.size(); i++) {
            grad.noalias() += transform_map[i].grad(time, state, param, noise_grad[i]);
        }
        return(grad);
    }
    inline vec sample(double time, const vec& state, const vec&param, std::mt19937* rng) {
        // compute noise params
        std::vector<vec> noise_params = compute_noise_params(time, state, param);
        // sample from the noise model
        vec obs = noise_model->sample(noise_params, rng);
        return(obs);
    }
    inline vec sample_np(double time, const vec& state, const vec& param, unsigned seed) {
        std::mt19937 rng(seed);
        return(sample(time, state, param, &rng));
    }
    inline vec transform(double time, const vec&state, const vec&param, const std::string& name) {
        unsigned ind = transform_index(name);
        return(transform_map[ind].transform(time, state, param));
    }
    // vectorized main functions
    inline vec log_prob_vec(double time, const vec& param, const vec& obs) {
        // iterate over states
        vec llh(num_states);
        for (int i = 0; i < num_states; i++) {
            llh[i] = log_prob(time, state_map[i], param, obs);
        }
        return(llh);
    } 
    inline mat_rm log_prob_grad_vec(double time, const vec& param, const vec& obs) {
        // iterate over states
        mat_rm llh_grad(num_states, param.size());
        for (int i = 0; i < num_states; i++) {
            llh_grad.row(i).noalias() = log_prob_grad(time, state_map[i], param, obs).transpose();
        }
        return(llh_grad);
    }
    inline mat_rm transform_vec(double time, const vec&param, const std::string& name) {
        unsigned ind = transform_index(name);
        unsigned dim = transform_map[ind].get_output_dim();
        mat_rm transformed(num_states, dim);
        for (unsigned i = 0; i < num_states; i++) {
            transformed.row(i).noalias() = transform_map[ind].transform(time, state_map[i], param).transpose();
        }
        return(transformed);
    }

    private:
    MJP* transition_model;
    std::string noise_type;
    NoiseModel* noise_model;
    std::vector<std::string> noise_param_list;
    std::vector<std::string> param_list;
    std::vector<Param> param_map;
    unsigned num_param;
    std::vector<Transform> transform_map;
    vec param_array;
    unsigned num_states;
    unsigned obs_dim;
    std::vector<vec> state_map;

};
