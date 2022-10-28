#pragma once

#include "../../types.h"

class Param {
    public:
    // constructor
    Param(const std::string& name_, const vec& value_);

    // getters
    inline const std::string& get_name() {
        return(name);
    }
    inline const vec& get_value() {
        return(value);
    }
    inline unsigned get_dim() {
        return(dim);
    }

    // setters
    void set_name(const std::string& name_) {
        name = name_;
    }

    protected:
    std::string name;
    vec value;
    unsigned dim;
    
};