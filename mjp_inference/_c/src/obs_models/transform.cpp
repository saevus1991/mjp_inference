#include "transform.h"

// constructor 

Transform::Transform(const std::string& name_, pybind11::tuple transform_callable, unsigned output_dim_, pybind11::tuple grad_callable) :
    name(name_),
    transform_fun(ut::get_pyfunction<TransformFun>(transform_callable)),
    output_dim(output_dim_),
    grad_fun(ut::get_pyfunction<TransformGrad>(grad_callable))
    {} 