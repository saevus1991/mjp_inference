#pragma once

#include "../../types.h"

class Param {
    public:
    // constructor
    Param(const std::string& name_, const vec& value_);
    Param(const std::string& name_, double value_);

    // getters
    inline const std::string& get_name() const {
        return(name);
    }
    inline const vec& get_value() const {
        return(value);
    }
    inline unsigned get_dim() const {
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