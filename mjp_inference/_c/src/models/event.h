#pragma once

#include <map>

#include "../../types.h"
#include "species.h"
#include "rate.h"
#include "../util/conversion.h"
#include "../util/misc.h"


class Event {

    public:
    // constructor
    Event(std::string name_, std::vector<std::string> input_species_, std::vector<std::string> output_species_, Rate rate_, pybind11::tuple hazard_callable_, std::vector<int> change_vec_, pybind11::dict species_dict_);
    Event(std::string name_, std::vector<std::string> input_species_, std::vector<std::string> output_species_, Rate rate_, pybind11::tuple hazard_callable_, std::vector<int> change_vec_, std::vector<Species*> species_map_);

    // helpers
    std::vector<std::string> build_species_list(std::vector<Species*>& species_map_);
    void assert_order();

    // getter
    inline const std::string& get_name() const {
        return(name);
    }
    inline const std::vector<std::string>& get_input_species() const {
        return(input_species);
    }
    inline const std::vector<std::string>& get_output_species() const {
        return(output_species);
    }
    inline const Rate& get_rate() const {
        return(rate);
    }
    inline ArrayFun get_hazard_fun() {
        return(hazard_fun);
    }
    inline const std::vector<int>& get_change_vec() const {
        return(change_vec);
    }
    inline std::vector<Species*>& get_species_map() {
        return(species_map);
    }
    inline pybind11::dict get_species_dict() {
        return(ut::vec2dict(species_list, species_map));
    }
    inline unsigned species_index(std::string species) {
        auto it = std::find(species_list.begin(), species_list.end(), species);
        unsigned ind = it - species_list.begin();
        return(ind);
    }
    inline Species* get_species_pointer(std::string species) {
        unsigned ind = species_index(species);
        return(species_map[ind]);
    }

    // setters
    inline void set_name(std::string name_) {
        name = name_;
    }
    inline void set_rate(const Rate& rate_) {
        rate = rate_;
    }
    inline void set_rate(double rate_) {
        rate.set_value(rate_);
    }
    inline void set_species_map(pybind11::dict species_dict) {
        species_list = ut::extract_dict_keys<std::string>(species_dict);
        species_map = ut::extract_dict_values<Species*>(species_dict);
        assert_order();
    }
    inline void set_species_map(std::vector<Species*> species_map_) {
        species_map = species_map_;
        species_list = build_species_list(species_map_);
        assert_order();
    }

    // main functions #TODO: make consistent with hazard and propensity
    inline double hazard(double* state) {
        return(rate.get_value() * hazard_fun(state));
    }
    inline double hazard_np(np_array_c state) {
        return(hazard((double*)state.data()));
    }

    private:
    std::string name;
    std::vector<std::string> input_species;
    std::vector<std::string> output_species;
    Rate rate;
    pybind11::tuple hazard_callable;
    pybind11::capsule hazard_capsule;
    ArrayFun hazard_fun;
    std::vector<int> change_vec;
    std::vector<std::string> species_list;
    std::vector<Species*> species_map;

};
