#include "event.h"

// constructor

Event::Event(std::string name_, std::vector<std::string> input_species_, std::vector<std::string> output_species_, Rate rate_, pybind11::tuple propensity_callable_, std::vector<int> change_vec_, pybind11::dict species_dict_) :
    name(name_),
    input_species(input_species_),
    output_species(output_species_),
    rate(rate_),
    propensity_callable(propensity_callable_),
    propensity_capsule(propensity_callable[0]),
    propensity_fun(reinterpret_cast<ArrayFun>(propensity_capsule.get_pointer())),
    change_vec(change_vec_),
    species_list(ut::extract_dict_keys<std::string>(species_dict_)),
    species_map(ut::extract_dict_values<Species*>(species_dict_))
    {
        if (species_list.size() > 0) {
            assert_order();
        }
    }

Event::Event(std::string name_, std::vector<std::string> input_species_, std::vector<std::string> output_species_, Rate rate_, pybind11::tuple propensity_callable_, std::vector<int> change_vec_, std::vector<Species*> species_map_) :
    name(name_),
    input_species(input_species_),
    output_species(output_species_),
    rate(rate_),
    propensity_callable(propensity_callable_),
    propensity_capsule(propensity_callable[0]),
    propensity_fun(reinterpret_cast<ArrayFun>(propensity_capsule.get_pointer())), 
    change_vec(change_vec_), 
    species_list(build_species_list(species_map_)),
    species_map(species_map_)
    {
        if (species_list.size() > 0) {
            assert_order();
        }
    }

// helpers

std::vector<std::string> Event::build_species_list(std::vector<Species*>& species_map_) {
    std::vector<std::string> species_list_(species_map_.size());
    for (unsigned i = 0; i < species_map_.size(); i++) {
        species_list_[i] = species_map_[i]->get_name();
    }
    return(species_list_); 
}

void Event::assert_order() {
    int index = -1;
    // check that input species are ordered according to species map
    for (int i = 0; i < input_species.size(); i++) {
        int species_ind = species_index(input_species[i]);;
        if (species_ind <= index) {
            std::string msg = "Make sure input species are used in the order induced by the model";
            throw std::invalid_argument(msg);
        }
        index = species_ind;
    }
    // reorder output species according to species list
    std::vector<unsigned> order(output_species.size());
    for (unsigned i = 0; i < output_species.size(); i++) {
        order[i] = species_index(output_species[i]);
    }
    output_species = ut::misc::sort_by(output_species, order);
    change_vec = ut::misc::sort_by(change_vec, order);
}

