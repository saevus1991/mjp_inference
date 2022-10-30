#include "mjp.h"

// constructor

MJP::MJP(std::string name_) : 
    name(name_),
    num_species(0),
    num_events(0),
    num_rates(0)
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
        // change vecs
        change_vectors.push_back(event_map[i].get_change_vec());
    }
    // extract rates if not provided
    if( rate_map.size() == 0 ) {
        for (int i = 0; i < num_events; i++) {
            const std::string& rate_name = event_map[i].get_rate().get_name();
            if (std::find(rate_list.begin(), rate_list.end(), rate_name) == rate_list.end()) {
                rate_list.push_back(rate_name);
                rate_map.push_back(event_map[i].get_rate());
            }
        }
        num_rates = rate_list.size();
    }
    // make event to rate map
    for (int i = 0; i < num_events; i++) {
        const std::string& rate_name = event_map[i].get_rate().get_name();
        unsigned ind = std::find(rate_list.begin(), rate_list.end(), rate_name) - rate_list.begin();
        event_to_rate_map.push_back(ind);
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

unsigned MJP::rate_index(const std::string& rate) {
    auto it = std::find(rate_list.begin(), rate_list.end(), rate);
    unsigned index = it - rate_list.begin();
    if (index == rate_list.size()) {
        std::string msg = "Rate \"" + rate + "\" not part of model \"" + name + "\"";
        throw std::invalid_argument(msg);
    }
    return(index);
}

mat_rm MJP::build_state_matrix() {
    mat_rm state_map(num_states, num_species);
    for (unsigned i = 0; i < num_states; i++) {
        state_map.row(i) = ind2state(i).transpose();
    }
    return(state_map);
}

std::vector<vec> MJP::build_state_map() {
    std::vector<vec> state_map(num_states);
    for (unsigned i = 0; i < num_states; i++) {
        state_map[i] = ind2state(i);
    }
    return(state_map);
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

void MJP::hazard(double* state, double* rates, double* haz) {
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
        double rate = rates[event_to_rate_map[i]];
        haz[i] = rate * event_map[i].propensity(local_state.data());
    }
}

vec MJP::hazard(vec& state, vec& rates) {
    vec haz(num_events);
    hazard(state.data(), rates.data(), haz.data());
    return(haz);
    // return(rates(event_to_rate_map).array() * propensity(state).array()); 
}

void MJP::propensity(double* state, double* prop) {
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
        prop[i] = event_map[i].propensity(local_state.data());
    }
}

vec MJP::propensity(vec& state) {
    vec prop(num_events);
    propensity(state.data(), prop.data());
    return(prop);
}

vec MJP::propensity(Eigen::Map<vec>& state) {
    vec prop(num_events);
    propensity(state.data(), prop.data());
    return(prop);
}

np_array MJP::propensity_out(np_array_c state) {
    np_array prop(num_events);
    propensity((double*)state.data(), (double*)prop.data());
    return(prop);
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

vec MJP::ind2state(unsigned ind) {
    std::vector<double> state = ut::lin2state<double, int>(ind, dims);
    return(ut::vec2vec(state));
}

np_array MJP::ind2state_np(unsigned ind) {
    std::vector<double> state = ut::lin2state<double, int>(ind, dims);
    return(ut::vec2array(state));
}

int MJP::state2ind(np_array state_) {
    std::vector<double> state((double*)state_.data(), (double*)state_.data() + state_.size());
    return(ut::state2lin(state, dims));
}