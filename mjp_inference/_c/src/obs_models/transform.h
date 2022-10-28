#pragma once

#include "../../types.h"
#include "../util/conversion.h"

class Transform {
    public:
    // constructor
    Transform(const std::string& name_, pybind11::tuple transform_callable, unsigned output_dim_, pybind11::tuple grad_state_callable, pybind11::tuple grad_param_callable);

    // getters
    inline const std::string& get_name() {
        return(name);
    }
    inline unsigned get_output_dim() {
        return(output_dim);
    }

    // main function
    inline vec transform(double time, vec& state, vec& param) {
        vec output(output_dim);
        transform_fun(time, state.data(), param.data(), output.data());
        return(output);
    }
    inline vec grad_state(double time, vec& state, vec&param, vec& grad_output) {
        vec grad(state.size());
        grad_state_fun(time, state.data(), param.data(), grad_output.data(), grad.data());
        return(grad);
    }
    inline vec grad_param(double time, vec& state, vec&param, vec& grad_output) {
        vec grad(param.size());
        grad_param_fun(time, state.data(), param.data(), grad_output.data(), grad.data());
        return(grad);
    }

    private:
    std::string name;
    TransformFun transform_fun;
    TransformGrad grad_state_fun;
    TransformGrad grad_param_fun;
    unsigned output_dim;
};