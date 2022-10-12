#pragma once

#include "../../types.h"

class Rate {
    public:

    // constructor
    Rate(const std::string& name_, double value_);

    // getters
    inline const std::string& get_name() const {
        return(name);
    }
    inline double get_value() const {
        return(value);
    }

    // setters
    inline void set_name(const std::string& name_) {
        name = name_;
    }
    inline void set_value(double value_) {
        value = value_;
    }

    private:
    std::string name;
    double value;
};