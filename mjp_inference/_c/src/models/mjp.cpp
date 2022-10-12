#include "mjp.h"

// constructor

MJP::MJP(std::string name_) : 
    name(name_),
    num_species(0),
    num_events(0)
    {}

// setup helper
void MJP::build() {
    /* Call this function after all events and species have been added */
    // construct dimensions
    dims = std::vector<int>(num_species);
    for (int i = 0; i < num_species; i++) {
        dims[i] = species_map[i].get_dim();
    }
    // set up default state
    default_state = std::vector<double>(num_species);
    for (int i = 0; i < num_species; i++) {
        default_state[i] = species_map[i].get_default();
    }
    // derive num_states
    num_states = 1;
    for (int i = 0; i < num_species; i++) {
        num_states *= dims[i];
    }
    // create event-related lists
    for (int i = 0; i < num_events; i++) {
        // input list
        std::vector<std::string> species_loc = event_map[i].get_input_species();
        input_species.push_back(species_index(species_loc));
        // output list
        species_loc = event_map[i].get_output_species();
        output_species.push_back(species_index(species_loc));
        // hazard funs
        hazard_funs.push_back(event_map[i].get_hazard_fun());
        // change vecs
        change_vectors.push_back(event_map[i].get_change_vec());
    }
}

// other helper functions

unsigned MJP::species_index(const std::string& species) {
    auto it = std::find(species_list.begin(), species_list.end(), species);
    unsigned index = it - species_list.begin();
    if (index == num_species) {
        std::string msg = "Species \"" + species + "\" not part of model \"" + name + "\"";
        throw std::invalid_argument(msg);
    }
    return(index);
}

std::vector<unsigned> MJP::species_index(const std::vector<std::string>& species) {
    unsigned size = species.size();
    std::vector<unsigned> index(size);
    for (int i = 0; i < size; i++) {
        index[i] = species_index(species[i]);
    }
    return(index);
}

unsigned MJP::event_index(const std::string& event) {
    auto it = std::find(event_list.begin(), event_list.end(), event);
    unsigned index = it - event_list.begin();
    if (index == num_events) {
        std::string msg = "Event \"" + event + "\" not part of model \"" + name + "\"";
        throw std::invalid_argument(msg);
    }
    return(index);
}

std::vector<unsigned> MJP::event_index(const std::vector<std::string>& event) {
    unsigned size = event.size();
    std::vector<unsigned> index(size);
    for (int i = 0; i < size; i++) {
        index[i] = event_index(event[i]);
    }
    return(index);
}

std::vector<std::vector<unsigned>> MJP::parse_clusters(std::vector<std::vector<std::string>> clusters) {
    std::vector<std::vector<unsigned>> converted_clusters(clusters.size());
    for (int i = 0; i < clusters.size(); i++) {
        converted_clusters[i] = species_index(clusters[i]);
    }
    return(converted_clusters);
}

// main functions

void MJP::hazard(double* state, double* haz) {
    // iterate over events
    for (int i = 0; i < num_events; i++) {
        // create local state
        unsigned local_size = input_species[i].size();
        std::vector<double> local_state(local_size);
        for (int j = 0; j < local_size; j++) {
            unsigned ind = input_species[i][j];
            local_state[j] = state[ind];
        }
        // eval local hazard 
        // haz[i] = hazard_funs[i](local_state.data());
        haz[i] = event_map[i].hazard(local_state.data());
    }
}

vec MJP::hazard(vec& state) {
    vec haz(num_events);
    hazard(state.data(), haz.data());
    return(haz);
}

vec MJP::hazard(Eigen::Map<vec>& state) {
    vec haz(num_events);
    hazard(state.data(), haz.data());
    return(haz);
}

np_array MJP::hazard_out(np_array_c state) {
    np_array haz(num_events);
    hazard((double*)state.data(), (double*)haz.data());
    return(haz);
}

void MJP::update_state(double* state, unsigned event) {
    for (int i = 0; i < output_species[event].size(); i++) {
        unsigned index = output_species[event][i];
        int change = change_vectors[event][i];
        state[index] += change;
    }
}

void MJP::update_state(vec&state, unsigned event) {
    update_state(state.data(), event); 
}

np_array MJP::update_state_out(np_array_c state, unsigned event) {
    np_array new_state(state.size(), (double*) state.data());
    update_state((double*) state.data(), event);
    return(new_state);
}

bool MJP::is_valid_state(std::vector<double>& state) {
    bool is_valid = true;
    for (int i = 0; i < num_species; i++) {
        if ( !species_map[i].is_valid_state(int(state[i])) ) {
            is_valid = false;
            break; 
        }
    }
    return(is_valid);
}

np_array MJP::ind2state(int ind) {
    std::vector<double> state = ut::lin2state<double, int>(ind, dims);
    return(ut::vec2array(state));
}

int MJP::state2ind(np_array state_) {
    std::vector<double> state((double*)state_.data(), (double*)state_.data() + state_.size());
    return(ut::state2lin(state, dims));
}