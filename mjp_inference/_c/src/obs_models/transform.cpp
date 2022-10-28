#include "transform.h"

// constructor 

Transform::Transform(const std::string& name_, pybind11::tuple transform_callable, unsigned output_dim_, pybind11::tuple grad_state_callable, pybind11::tuple grad_param_callable) :
    name(name_),
    transform_fun(ut::get_pyfunction<TransformFun>(transform_callable)),
    output_dim(output_dim_),
    grad_state_fun(ut::get_pyfunction<TransformGrad>(grad_state_callable)),
    grad_param_fun(ut::get_pyfunction<TransformGrad>(grad_param_callable))
    {} 