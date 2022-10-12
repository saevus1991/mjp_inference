#pragma once

#include "../../types.h" 


class Species {

    public:
    Species(std::string name_, int lower_, int upper_, int default_value_);

    // getters
    inline std::string get_name() {
        return(name);
    }
    inline int get_lower() {
        return(lower);
    }
    inline int get_upper() {
        return(upper);
    }
    inline int get_default() {
        return(default_value);
    }
    inline int get_index() {
        return(index);
    }
    inline int get_dim() {
        return(upper-lower+1);
    }

    // setters
    void set_name(std::string& name_) {
        name = name_;
    }
    void set_lower(int lower_) {
        lower = lower_;
    }
    void set_upper(int upper_) {
        upper = upper_;
    }
    void set_default(int default_value_) {
        default_value = default_value_;
    }
    void set_index(int index_) {
        index = index_;
    }

    // main functions
    inline bool is_valid_state(int state) {
        if (state >= lower && state <= upper) {
            return(true);
        } else {
            return(false);
        }
    }

    private:
    std::string name;
    int lower;
    int upper;
    int default_value;
    int index;

};