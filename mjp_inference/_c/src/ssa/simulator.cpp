#include "simulator.h"

// constructor

Simulator::Simulator(MJP* model_, vec initial_, vec tspan_, std::mt19937* rng_, int max_events_, std::string max_event_handler_) :
    model(model_),
    initial(initial_),
    rates(model->get_rate_array()),
    tspan(tspan_),
    rng(rng_),
    max_events(max_events_),
    max_event_handler(max_event_handler_)
    {}

Simulator::Simulator(MJP* model_, int num_states, vec rates_, vec tspan_, std::mt19937* rng_, int max_events_in, std::string max_event_handler_) :
    model(model_),
    initial(num_states),
    rates(rates_),
    tspan(tspan_),
    rng(rng_),
    max_events(max_events_in),
    max_event_handler(max_event_handler_)
    {}

Simulator::Simulator(MJP* model_, vec initial_, vec rates_, vec tspan_, std::mt19937* rng_, int max_events_, std::string max_event_handler_) :
    model(model_),
    initial(initial_),
    rates(rates_),
    tspan(tspan_),
    rng(rng_),
    max_events(max_events_),
    max_event_handler(max_event_handler_)
    {}


// main functions

Trajectory Simulator::simulate() {
    // create storage vectors
    Trajectory trajectory;
    trajectory.tspan = tspan;
    trajectory.initial = initial;
    trajectory.time.reserve(1000);
    trajectory.events.reserve(1000);
    // preprations
    double t = tspan[0];
    double t_max = tspan[1];
    vec state = initial;
    vec hazard(model->get_num_events());
    int num_events = 0;
    // create sample path
    while (num_events < max_events && t < t_max) {
        // determine next event
        hazard.noalias() = model->hazard(state, rates);
        std::tuple<bool, double, int> reaction_triplet = next_reaction(t, hazard);
        if (!std::get<0>(reaction_triplet)) {
            break;
        }
        // update system
        t += std::get<1>(reaction_triplet);
        model->update_state(state, std::get<2>(reaction_triplet));
        // update output statistics
        trajectory.events.push_back(std::get<2>(reaction_triplet));
        trajectory.time.push_back(t);
        num_events++;
    }
    if (num_events >= max_events) {
        if (max_event_handler == "warning") 
            pybind11::print("Warning: Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.");
        else if (max_event_handler == "error") {
            std::string msg = "Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.";
            throw std::runtime_error(msg);
        }
    };
    // build trajectory
    trajectory.states = Trajectory::construct_trajectory(model, trajectory.events, initial);
    return(trajectory);
}

pybind11::dict Simulator::simulate_py() {
    return(simulate().to_dict());
}


np_array Simulator::simulate(np_array_c t_eval) {
    // preprations
    double* t_eval_ptr = (double*) t_eval.data();
    int num_steps = t_eval.size();
    int num_species = model->get_num_species();
    double t = tspan[0];
    double t_max = tspan[1];
    vec state = initial;
    double* state_ptr = (double*) state.data();
    vec hazard(model->get_num_events());         
    // double* hazard_ptr = hazard.data_ptr<double>();
    int num_events = 0;
    int t_eval_index = 0;
    // create output
    np_array states(std::vector<int>({num_steps, num_species}));
    double* states_ptr = (double*) states.data();
    // create sample path
    while (num_events < max_events && t < t_max) {
        // determine next event
        hazard.noalias() = model->hazard(state, rates);
        std::tuple<bool, double, int> reaction_triplet = next_reaction(t, hazard);
        if (!std::get<0>(reaction_triplet)) {
            break;
        }
        // update time
        t += std::get<1>(reaction_triplet);
        // update states
        while (t_eval_index < num_steps && t > t_eval_ptr[t_eval_index]) {
            for (int j = 0; j < num_species; j++) {
                states_ptr[t_eval_index*num_species+j] = state_ptr[j];
            }
            t_eval_index++;
        }
        // update system
        model->update_state(state, std::get<2>(reaction_triplet));
        num_events++;
    }
    // update up to final time
    while (t_eval_index < num_steps) {
        for (int j = 0; j < num_species; j++) {
            states_ptr[t_eval_index*num_species+j] = state_ptr[j];
        }
        t_eval_index++;
    }
    if (num_events >= max_events) {
        if (max_event_handler == "warning") 
            pybind11::print("Warning: Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.");
        else if (max_event_handler == "error") {
            std::string msg = "Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.";
            throw std::runtime_error(msg);
        }
    };
    // produce python output
    return(states);
}


mat_rm Simulator::simulate(const Eigen::Map<vec>& t_eval) {
    // preprations
    const double* t_eval_ptr = t_eval.data();
    int num_steps = t_eval.size();
    int num_species = model->get_num_species();
    double t = tspan[0];
    double t_max = tspan[1];
    vec state = initial;
    double* state_ptr = (double*) state.data();
    vec hazard(model->get_num_events());       
    int num_events = 0;
    int t_eval_index = 0;
    // create output
    mat_rm states(num_steps, num_species);
    double* states_ptr = states.data();
    // create sample path
    while (num_events < max_events && t < t_max) {
        // determine next event
        hazard.noalias() = model->hazard(state, rates);
        std::tuple<bool, double, int> reaction_triplet = next_reaction(t, hazard);
        if (!std::get<0>(reaction_triplet)) {
            break;
        }
        // update time
        t += std::get<1>(reaction_triplet);
        // update states
        while (t_eval_index < num_steps && t > t_eval_ptr[t_eval_index]) {
            for (int j = 0; j < num_species; j++) {
                states_ptr[t_eval_index*num_species+j] = state_ptr[j];
            }
            t_eval_index++;
        }
        // update system
        model->update_state(state, std::get<2>(reaction_triplet));
        num_events++;
    }
    // update up to final time
    while (t_eval_index < num_steps) {
        for (int j = 0; j < num_species; j++) {
            states_ptr[t_eval_index*num_species+j] = state_ptr[j];
        }
        t_eval_index++;
    }
    if (num_events >= max_events) {
        if (max_event_handler == "warning") 
            pybind11::print("Warning: Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.");
        else if (max_event_handler == "error") {
            std::string msg = "Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.";
            throw std::runtime_error(msg);
        }
    };
    // produce python output
    return(states);
}


std::tuple<bool, double, int> Simulator::next_reaction(double time, vec& hazard) {
    /* Calculates reaction times for all channels. The mimimum time and the corresponding index are saved in delta_t and index. */
    // calculate the reaction hazards
    vec cum_hazard = ut::math::cumsum(hazard);
    double total_hazard = cum_hazard(Eigen::last);
    // create random numbers
    double rand_2 = U(*rng);
    double rand_1 = U(*rng);
    // check if reaction happens
    int index = 0;
    bool fired = false;
    double delta_t = -std::log(rand_1)/total_hazard;
    if (total_hazard > 0.0 && time + delta_t < tspan[1]) {
        fired = true;
    }
    // perform updates
    if (fired) {
        // sample random event from the individual hazards
        rand_2 *= total_hazard;
        double* cum_hazard_ptr = cum_hazard.data();
        while ( cum_hazard_ptr[index] < rand_2) {
            index++;
        }
    }
    return(std::tuple<bool, double, int>(fired, delta_t, index));
}

// #TODO: add a post simulation discretization

// *** wrapper class *** 

PySimulator::PySimulator(MJP* model_, vec initial_, vec rates_, vec tspan_, int seed_, int max_events_, std::string max_event_handler_) :
    Simulator(model_, initial_, rates_, tspan_, nullptr, max_events_, max_event_handler_),
    rng_raw(seed_) {
        rng = &rng_raw;
    }

PySimulator::PySimulator(MJP* model_, vec initial_, vec tspan_, int seed_, int max_events_, std::string max_event_handler_) : PySimulator(model_, initial_, model_->get_rate_array(), tspan_, seed_, max_events_, max_event_handler_) {}