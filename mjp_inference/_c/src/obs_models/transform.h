#pragma once

#include "../../types.h"
#include "../util/conversion.h"

class Transform {
    public:
    // constructor
    Transform(const std::string& name_, pybind11::tuple transform_callable, unsigned output_dim_, pybind11::tuple grad_callable);

    // getters
    inline const std::string& get_name() const {
        return(name);
    }
    inline unsigned get_output_dim() const {
        return(output_dim);
    }

    // main function
    inline vec transform(double time, const vec& state, const vec& param) const {
        vec output(output_dim);
        transform_fun(time, state.data(), param.data(), output.data());
        return(output);
    }
    inline vec grad(double time, const vec& state, const vec&param, const vec& grad_output) const {
        vec grad(param.size());
        grad.setZero();
        grad_fun(time, state.data(), param.data(), grad_output.data(), grad.data());
        return(grad);
    }

    private:
    std::string name;
    TransformFun transform_fun;
    TransformGrad grad_fun;
    unsigned output_dim;
};