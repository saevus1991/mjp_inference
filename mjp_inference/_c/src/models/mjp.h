#pragma once

#include <map>
#include <algorithm>

#include "../../types.h"
#include "species.h"
#include "event.h"
#include "../util/conversion.h"


class MJP {
    public:
    // constructor
    MJP(std::string name_);

    // setup helper
    void build();

    // other helper functions
    unsigned species_index(const std::string& species);
    std::vector<unsigned> species_index(const std::vector<std::string>& species);
    unsigned event_index(const std::string& event);
    std::vector<unsigned> event_index(const std::vector<std::string>& event);
    std::vector<std::vector<unsigned>> parse_clusters(std::vector<std::vector<std::string>> clusters);

    // interface functions
    inline void add_species(Species species) {
        species_map.push_back(species);
        species_list.push_back(species.get_name());
        num_species++;
    }
    inline void make_add_species(std::string name, int lower, int upper, int default_value) {
        add_species(Species(name, lower, upper, default_value));
    }
    inline void add_event(Event event) {
        if (event.get_species_map().size() == 0) {
            event.set_species_map(get_species_pointers());
        }
        event_map.push_back(event);
        event_list.push_back(event.get_name());
        num_events++;
    }
    inline void make_add_event(std::string name, std::vector<std::string> input_species, std::vector<std::string> output_species, double rate_, pybind11::tuple hazard_callable, std::vector<int> change_vec) {
        Rate rate(name, rate_);
        add_event(Event(name, input_species, output_species, rate, hazard_callable, change_vec, get_species_pointers()));
    }

    // getters
    inline const std::string& get_name() const{
        return(name);
    }
    inline unsigned get_num_species() {
        return(num_species);
    }
    inline unsigned get_num_events() {
        return(num_events);
    }
    inline unsigned get_num_states() {
        return(num_states);
    }
    inline const std::vector<int>& get_dims() const{
        return(dims);
    }
    inline const Species& get_species(unsigned ind) {
        return(species_map[ind]);
    }
    inline const Species& get_species(const std::string& species) {
        return(get_species(species_index(species)));
    }
    inline const Event& get_event(unsigned ind) {
        return(event_map[ind]);
    }
    inline const Event& get_event(const std::string& event) {
        return(get_event(event_index(event)));
    }
    inline std::vector<unsigned> get_local_dims(const std::vector<unsigned>& node_list) {
        std::vector<unsigned> local_dims(node_list.size());
        for (int i = 0; i < node_list.size(); i++) {
            local_dims[i] = dims[node_list[i]];
        }
        return(local_dims);
    }
    inline std::vector<double> get_default_state() {
        return(default_state);
    }
    inline pybind11::dict get_species_dict() {
        return(ut::vec2pointerdict<std::string, Species>(species_list, species_map));
    }
    inline pybind11::dict get_event_dict() {
        return(ut::vec2pointerdict<std::string, Event>(event_list, event_map));
    }
    inline std::vector<std::string> get_species_list() {
        return(species_list);
    }
    inline std::vector<std::string> get_event_list() {
        return(event_list);
    }
    inline const std::vector<std::string>& get_rate_list () const {
        return(rate_list);
    }
    inline std::vector<Species>& get_species_map() {
        return(species_map);
    }
    inline std::vector<Event>& get_event_map() {
        return(event_map);
    }
    inline std::vector<Species*> get_species_pointers() {
        std::vector<Species*> species_pointers(species_map.size());
        for (unsigned i = 0; i < species_map.size(); i++) {
            species_pointers[i] = &species_map[i];
        }
        return(species_pointers);
    } 
    std::vector<std::vector<unsigned>>& get_input_species() {
        return(input_species);
    }
    std::vector<std::vector<unsigned>>& get_output_species() {
        return(output_species);
    }
    std::vector<std::vector<int>>& get_change_vectors() {
        return(change_vectors);
    }

    // setters
    inline void set_name(std::string name_) {
        name = name_;
    }

    // main functions
    void hazard(double* state, double* haz);
    vec hazard(vec& state);
    vec hazard(Eigen::Map<vec>& state);
    np_array hazard_out(np_array_c state);
    void update_state(double* state, unsigned event);
    void update_state(vec&state, unsigned event);
    np_array update_state_out(np_array_c state, unsigned event);
    bool is_valid_state(std::vector<double>& state);
    np_array ind2state(int ind);
    int state2ind(np_array state);

    private:
    std::string name;
    std::vector<Species> species_map;
    std::vector<Event> event_map;
    unsigned num_species;
    unsigned num_events;
    unsigned num_states;
    std::vector<std::string> species_list;
    std::vector<std::string> event_list;
    std::vector<std::string> rate_list;
    std::vector<unsigned> event_to_rate_map;
    std::vector<int> dims;
    std::vector<double> default_state;
    std::vector<std::vector<unsigned>> input_species;
    std::vector<std::vector<unsigned>> output_species;
    std::vector<std::vector<int>> change_vectors;
    std::vector<ArrayFun> hazard_funs;

};


