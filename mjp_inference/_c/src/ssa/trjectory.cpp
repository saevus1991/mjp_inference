#include "trajectory.h"

// construct from dict

Trajectory::Trajectory(pybind11::dict trajectory) {
    // convert initial
    if (trajectory.contains("initial")) {
        np_array initial_tmp = trajectory["initial"].cast<np_array_c>();
        initial = Eigen::Map<vec>((double*) initial_tmp.data(), initial_tmp.size());
    } else {
        std::string msg = "Trajectory must contain key initial";
        throw std::invalid_argument(msg);
    }
    // convert tspan
    if (trajectory.contains("tspan")) { 
        np_array tspan_tmp = trajectory["tspan"].cast<np_array_c>();
        tspan = Eigen::Map<vec>((double*) tspan_tmp.data(), tspan_tmp.size()); 
    } else {
        std::string msg = "Trajectory must contain key tspan";
        throw std::invalid_argument(msg);
    }
    // convert times
    if (trajectory.contains("times")) { 
        np_array times_tmp = trajectory["times"].cast<np_array_c>();
        double *times_ptr = (double*) times_tmp.data();
        time = std::vector<double>(times_ptr, times_ptr + times_tmp.size());
    } else {
        std::string msg = "Trajectory must contain key times";
        throw std::invalid_argument(msg);
    }
    if ( !trajectory.contains("events") && !trajectory.contains("states")) {
        std::string msg = "Trajectory must contain at least one of these keys: events, states";
        throw std::invalid_argument(msg);
    }
    // convert events
    if (trajectory.contains("events")) { 
        np_array events_tmp = trajectory["events"].cast<np_array_c>();
        events.reserve(events_tmp.size());
        double* events_ptr = (double*) events_tmp.size();
        for (int i = 0; i < events_tmp.size(); i++) {
            events.push_back(int(events_ptr[i]));
        }
    }
    // convert states
    if (trajectory.contains("states")) { 
        np_array states_tmp = trajectory["states"].cast<np_array_c>();
        states = Eigen::Map<mat_rm>((double*) states_tmp.data(), states_tmp.shape(0), states_tmp.shape(1));
    }
}


// helper functions

pybind11::dict Trajectory::to_dict() {
    pybind11::dict trajectory;
    trajectory["initial"] = ut::vec2array(initial);
    trajectory["tspan"] = ut::vec2array(tspan);
    trajectory["times"] = ut::vec2array(time);
    trajectory["events"] = ut::vec2array(events);
    trajectory["states"] = ut::mat2array(states);
    return(trajectory);
}

//  static functions


mat_rm Trajectory::construct_trajectory(MJP* model, const std::vector<int>& events, vec& initial) {
    // get initial state
    int num_species = model->get_num_species();
    mat_rm states(events.size(), num_species);
    double *states_ptr = (double*) states.data();
    vec state = initial;
    double *state_ptr = state.data();
    // fill up the trajectory by iterating over the events
    for (unsigned i = 0; i < events.size(); i++ ) {
        // update the state
        int index = events[i];
        model->update_state(state, index);
        // append to the vector
        for (unsigned j = 0; j < num_species; j++) {
            states_ptr[i*num_species+j] = state_ptr[j];
        }
    }
    return(states);
}