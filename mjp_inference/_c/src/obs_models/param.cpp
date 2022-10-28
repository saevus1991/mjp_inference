#include "param.h"

// constructor

Param::Param(const std::string& name_, const vec& value_) :
    name(name_),
    value(value_),
    dim(value.size())
    {}