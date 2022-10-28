#include "param.h"

// constructor

Param::Param(const std::string& name_, const vec& value_) :
    name(name_),
    value(value_),
    dim(value.size())
    {}

Param::Param(const std::string& name_, double value_) :
    name(name_),
    value(1),
    dim(1)
    {
        value[0] = value_;
    }
