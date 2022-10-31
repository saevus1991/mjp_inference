#include "posterior_simulator.h"

PosteriorSimulator::PosteriorSimulator(MJP* model_in, const vec& initial_in, const vec& tspan_in, int seed_in, int max_events_in, const std::string& max_event_handler_in) :
    model(model_in),
    initial(initial_in),
    rates(model->get_rate_array()),
    tspan(tspan_in),
    rng(seed_in),
    max_events(max_events_in),
    max_event_handler(max_event_handler_in)
    {}

PosteriorSimulator::PosteriorSimulator(MJP* model_in, int num_states, const vec& rates_in, const vec& tspan_in, int seed_in, int max_events_in, const std::string& max_event_handler_in) :
    model(model_in),
    initial(num_states),
    rates(rates_in),
    tspan(tspan_in),
    rng(seed_in),
    max_events(max_events_in),
    max_event_handler(max_event_handler_in)
    {}


template <class S, class T>
Trajectory PosteriorSimulator::simulate(S& t_grid, T& backward) {
    // set up output trajectory
    Trajectory trajectory;
    trajectory.time.reserve(1000);
    trajectory.events.reserve(1000);
    trajectory.initial = initial;
    trajectory.tspan = tspan;
    // preprations
    t_index = 0;
    double t = tspan[0];
    double t_max = tspan[1];
    vec state = initial;
    vec hazard(model->get_num_events());
    int num_events = 0;
    // set up vectors for internal time
    std::vector<double> internal_time(model->get_num_events());
    std::vector<double> stats(model->get_num_events());
    // initialize the internal times
    for (unsigned i = 0; i < model->get_num_events(); i++) {
        internal_time[i] = -std::log(U(rng));
    }
    // create sample path
    while (num_events < max_events && t < t_max) {
        // determine next event
        hazard.noalias() = model->hazard(state, rates);
        std::tuple<bool, double, int> reaction_triplet = next_reaction(t, state, hazard, t_grid, backward, internal_time, stats);
        if (!std::get<0>(reaction_triplet)) {
            break;
        }
        // update system
        t += std::get<1>(reaction_triplet);
        model->update_state(state, std::get<2>(reaction_triplet));;
        // update output statistics
        trajectory.events.push_back(std::get<2>(reaction_triplet));
        trajectory.time.push_back(t);
        num_events++;
        // update internal time for the reaction that fired
        internal_time[std::get<2>(reaction_triplet)] += -std::log(U(rng));
    }
    if (num_events >= max_events) {
        if (max_event_handler == "warning") 
            pybind11::print("Warning: Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.");
        else if (max_event_handler == "error") {
            std::string msg = "Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.";
            throw std::runtime_error(msg);
        }
    };
    // convert states
    trajectory.states = Trajectory::construct_trajectory(model, trajectory.events, initial);
    return(trajectory);
}

template Trajectory PosteriorSimulator::simulate<vec, mat_rm>(vec& t_grid, mat_rm& backward);
template Trajectory PosteriorSimulator::simulate<Eigen::Map<vec>, Eigen::Map<mat_rm>>(Eigen::Map<vec>& t_grid, Eigen::Map<mat_rm>& backward);
template Trajectory PosteriorSimulator::simulate<Eigen::Map<vec>, mat_rm>(Eigen::Map<vec>& t_grid, mat_rm& backward);

pybind11::dict PosteriorSimulator::simulate(np_array_c t_grid_in, np_array_c backward_in) {
    // parse iput
    Eigen::Map<vec> t_grid((double*) t_grid_in.data(), t_grid_in.size());
    Eigen::Map<mat_rm> backward((double*) backward_in.data(), backward_in.shape(0), backward_in.shape(1));
    // return output
    return(simulate(t_grid, backward).to_dict());
}

Trajectory PosteriorSimulator::simulate(Eigen::Map<vec>& t_grid, KrylovBackwardFilter& filt) {
    // set up output trajectory
    Trajectory trajectory;
    trajectory.time.reserve(1000);
    trajectory.events.reserve(1000);
    trajectory.initial = initial;
    trajectory.tspan = tspan;
    // preprations
    t_index = 0;
    double t = tspan[0];
    double t_max = tspan[1];
    vec state = initial;
    vec hazard(model->get_num_events());
    int num_events = 0;
    // set up vectors for internal time
    std::vector<double> internal_time(model->get_num_events());
    std::vector<double> stats(model->get_num_events());
    // initialize the internal times
    for (unsigned i = 0; i < model->get_num_events(); i++) {
        internal_time[i] = -std::log(U(rng));
    }
    // create sample path
    while (num_events < max_events && t < t_max) {
        // determine next event
        hazard.noalias() = model->hazard(state, rates);
        std::tuple<bool, double, int> reaction_triplet = next_reaction(t, state, hazard, t_grid, filt, internal_time, stats);
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
        // update internal time for the reaction that fired
        internal_time[std::get<2>(reaction_triplet)] += -std::log(U(rng));
    }
    if (num_events >= max_events) {
        if (max_event_handler == "warning") 
            pybind11::print("Warning: Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.");
        else if (max_event_handler == "error") {
            std::string msg = "Simulation stopped. Maximum number of " + std::to_string(max_events) + " of events exceeded.";
            throw std::runtime_error(msg);
        }
    };
    // convert states
    trajectory.states = Trajectory::construct_trajectory(model, trajectory.events, initial);
    return(trajectory);
}

// std::tuple<bool, double, int> PosteriorSimulator::next_reaction(double time, vec& hazard) {
//     /* Calculates reaction times for all channels. The mimimum time and the corresponding index are saved in delta_t and index. */
//     // calculate the reaction hazards
//     vec cum_hazard = ut::cumsum(hazard);
//     double total_hazard = cum_hazard(Eigen::last);
//     // create random numbers
//     double rand_2 = U(rng);
//     double rand_1 = U(rng);
//     // check if reaction happens
//     int index = 0;
//     bool fired = false;
//     double delta_t = -std::log(rand_1)/total_hazard;
//     if (total_hazard > 0.0 && time + delta_t < tspan[1]) {
//         fired = true;
//     }
//     // perform updates
//     if (fired) {
//         // sample random event from the individual hazards
//         rand_2 *= total_hazard;
//         double* cum_hazard_ptr = cum_hazard.data();
//         while ( cum_hazard_ptr[index] < rand_2) {
//             index++;
//         }
//     }
//     return(std::tuple<bool, double, int>(fired, delta_t, index));
// }

template <class S, class T>
std::tuple<bool, double, int> PosteriorSimulator::next_reaction (double time, vec& state, vec& hazard, S& time_grid, T& backward, std::vector<double>& internal_time, std::vector<double>& stats) {
    /* Calculates reaction times for all channels. The mimimum time and the corresponding index are saved in delta_t and index. */
    // preparations
    int index = -1;
    double delta_t;
    int num_reactions = model->get_num_events();
    int num_steps = time_grid.rows();
    int state_ind = model->state2ind(state);
    std::vector<int> targets = model->targets(state);
    // get
    vec backward_tmp = backward.row(t_index).transpose();
    vec control = get_control(state_ind, targets, backward_tmp);
    backward_tmp.noalias() = backward.row(t_index+1).transpose();
    vec next_control = get_control(state_ind, targets, backward_tmp);
    // initilize some variables for storing temporary information
    std::vector<double> tmp1(num_reactions,0.0);
    std::vector<double> tmp2(num_reactions,0.0);
    std::vector<double> tmp3(num_reactions,0.0);
    // initialize tmp2 and tmp3 by the integral from the current time to the next larger grid step
    for ( int i = 0; i < num_reactions; i++) {
        if (hazard[i] > 0) {
            tmp1[i] = (internal_time[i]-stats[i]) / hazard[i];
            tmp2[i] = (next_control[i]-control[i]) * (time-time_grid[t_index]) / (time_grid[t_index+1]-time_grid[t_index]);
            tmp2[i] = 0.5*(tmp2[i]+control[i]+next_control[i])*(time_grid[t_index+1]-time);
            tmp3[i] = tmp2[i];
        }
    }
    t_index++;
    bool fired = false;
    // check if a reaction has already fired in the initial interval
    for ( int i = 0; i < num_reactions; i++) {
        if ( hazard[i] > 0 && tmp2[i] > tmp1[i]) {
            fired = true;
            break;
        }
    }
    // iterate over all reactions and increase index until first reaction fires
    while ( t_index < num_steps-1 && !fired ) {
        // update controls
        backward_tmp.noalias() = backward.row(t_index).transpose();
        control = get_control(state_ind, targets, backward_tmp);
        backward_tmp.noalias() = backward.row(t_index+1).transpose();
        next_control = get_control(state_ind, targets, backward_tmp);
        // calculate the increment to the integral
        for ( int i = 0; i < num_reactions; i++) {
            // if propensity is positive, evaluate the next time
            if (hazard[i] > 0) {
                tmp3[i] = 0.5 * (next_control[i]+control[i]) * (time_grid[t_index+1]-time_grid[t_index]);
                tmp2[i] += tmp3[i];
            }
        }
        t_index++;
        // check if any reaction has fired
        for ( int i = 0; i <num_reactions; i++) {
            if ( hazard[i] > 0 && tmp2[i] > tmp1[i]) {
                fired = true;
                break;
            }
        }
    }
    // if a reaction has fired, undo the last update
    if ( fired ) {
        for ( int i = 0; i < num_reactions; i++ ) {
            if (hazard[i] > 0) {
                tmp2[i] -= tmp3[i];
            }
        }
        t_index--;
        // update the remainder term for the integral
        for (int i = 0; i < num_reactions; i++) {
            if (hazard[i] > 0) {
                tmp1[i] -= tmp2[i];
            }
        }
        // the remaining integral is quadratic and can be inverted analytically
        double min_time = max_double;
        backward_tmp.noalias() = backward.row(t_index).transpose();
        control = get_control(state_ind, targets, backward_tmp);
        backward_tmp.noalias() = backward.row(t_index+1).transpose();
        next_control = get_control(state_ind, targets, backward_tmp);
        for ( int i = 0; i < num_reactions; i++) {
            if (hazard[i] > 0 && tmp1[i] < tmp3[i]) { // the second condition ensures that a positive solution exists
                // solve for time
                double a = 0.5 * (next_control[i]-control[i]) / (time_grid[t_index+1]-time_grid[t_index]);
                double b = control[i] + 2 * a * std::max(0.0, time - time_grid[t_index]);
                double tmp_time = ut::math::solve_quadratic(a, b, -tmp1[i]);
                // store reaction index
                if ( tmp_time < min_time) {
                    min_time = tmp_time;
                    index = i;
                }
            }
        }
        // calculate the time of the next reaction
        delta_t = std::max(0.0, time_grid[t_index]-time) + min_time;
        // update the internal time for all the reactions
        for ( int i = 0; i < num_reactions; i++ ) {
            double a = 0.5*(next_control[i]-control[i])/(time_grid[t_index+1]-time_grid[t_index]);
            double b = control[i] + 2 * a * std::max(0.0, time - time_grid[t_index]);
            stats[i] += (tmp2[i]+min_time*(b+a*min_time)) * hazard[i];
        }
    }
    else { // update the statistics up to the final time
        for ( int i = 0; i < num_reactions; i++) {
            stats[i] += tmp2[i] * hazard[i];
        }
    }
    return(std::tuple<bool, double, int>(fired, delta_t, index));
}

template std::tuple<bool, double, int> PosteriorSimulator::next_reaction<Eigen::Map<vec>, Eigen::Map<mat_rm>> (double time, vec& state, vec& hazard, Eigen::Map<vec>& time_grid, Eigen::Map<mat_rm>& backward, std::vector<double>& internal_time, std::vector<double>& stats);

std::tuple<bool, double, int> PosteriorSimulator::next_reaction (double time, vec& state, vec& hazard, Eigen::Map<vec>& time_grid, KrylovBackwardFilter& filt, std::vector<double>& internal_time, std::vector<double>& stats) {
    /* Calculates reaction times for all channels. The mimimum time and the corresponding index are saved in delta_t and index. */
    // preparations
    int index = -1;
    double delta_t;
    unsigned num_reactions = model->get_num_events();
    unsigned num_steps = time_grid.rows();
    unsigned state_ind = model->state2ind(state);
    std::vector<int> targets = model->targets(state);
    // get
    vec backward_tmp = filt.eval_backward_filter(time_grid[t_index]);
    vec control = get_control(state_ind, targets, backward_tmp);
    backward_tmp.noalias() = filt.eval_backward_filter(time_grid[t_index+1]);
    vec next_control = get_control(state_ind, targets, backward_tmp);
    // initilize some variables for storing temporary information
    std::vector<double> tmp1(num_reactions,0.0);
    std::vector<double> tmp2(num_reactions,0.0);
    std::vector<double> tmp3(num_reactions,0.0);
    // initialize tmp2 and tmp3 by the integral from the current time to the next larger grid step
    for ( int i = 0; i < num_reactions; i++) {
        if (hazard[i] > 0) {
            tmp1[i] = (internal_time[i]-stats[i]) / hazard[i];
            tmp2[i] = (next_control[i]-control[i]) * (time-time_grid[t_index]) / (time_grid[t_index+1]-time_grid[t_index]);
            tmp2[i] = 0.5*(tmp2[i]+control[i]+next_control[i])*(time_grid[t_index+1]-time);
            tmp3[i] = tmp2[i];
        }
    }
    t_index++;
    bool fired = false;
    // check if a reaction has already fired in the initial interval
    for ( int i = 0; i < num_reactions; i++) {
        if ( hazard[i] > 0 && tmp2[i] > tmp1[i]) {
            fired = true;
            break;
        }
    }
    // iterate over all reactions and increase index until first reaction fires
    while ( t_index < num_steps-1 && !fired ) {
        // update controls
        backward_tmp.noalias() = filt.eval_backward_filter(time_grid[t_index]);
        control = get_control(state_ind, targets, backward_tmp);
        backward_tmp.noalias() = filt.eval_backward_filter(time_grid[t_index+1]);
        next_control = get_control(state_ind, targets, backward_tmp);
        // calculate the increment to the integral
        for ( int i = 0; i < num_reactions; i++) {
            // if propensity is positive, evaluate the next time
            if (hazard[i] > 0) {
                tmp3[i] = 0.5 * (next_control[i]+control[i]) * (time_grid[t_index+1]-time_grid[t_index]);
                tmp2[i] += tmp3[i];
            }
        }
        t_index++;
        // check if any reaction has fired
        for ( int i = 0; i <num_reactions; i++) {
            if ( hazard[i] > 0 && tmp2[i] > tmp1[i]) {
                fired = true;
                break;
            }
        }
    }
    // if a reaction has fired, undo the last update
    if ( fired ) {
        for ( int i = 0; i < num_reactions; i++ ) {
            if (hazard[i] > 0) {
                tmp2[i] -= tmp3[i];
            }
        }
        t_index--;
        // update the remainder term for the integral
        for (int i = 0; i < num_reactions; i++) {
            if (hazard[i] > 0) {
                tmp1[i] -= tmp2[i];
            }
        }
        // the remaining integral is quadratic and can be inverted analytically
        double min_time = tspan[1];
        backward_tmp.noalias() = filt.eval_backward_filter(time_grid[t_index]);
        control = get_control(state_ind, targets, backward_tmp);
        backward_tmp.noalias() = filt.eval_backward_filter(time_grid[t_index+1]);
        next_control = get_control(state_ind, targets, backward_tmp);
        for ( int i = 0; i < num_reactions; i++) {
            if (hazard[i] > 0 && tmp1[i] < tmp3[i]) { // the second condition ensures that a positive solution exists
                // solve for time
                double a = 0.5 * (next_control[i]-control[i]) / (time_grid[t_index+1]-time_grid[t_index]);
                double b = control[i] + 2 * a * std::max(0.0, time - time_grid[t_index]);
                double tmp_time = ut::math::solve_quadratic(a, b, -tmp1[i]);
                // store reaction index
                if ( tmp_time < min_time) {
                    min_time = tmp_time;
                    index = i;
                }
            }
        }
        // calculate the time of the next reaction
        delta_t = std::max(0.0, time_grid[t_index]-time) + min_time;
        // update the internal time for all the reactions
        for ( int i = 0; i < num_reactions; i++ ) {
            double a = 0.5*(next_control[i]-control[i])/(time_grid[t_index+1]-time_grid[t_index]);
            double b = control[i] + 2 * a * std::max(0.0, time - time_grid[t_index]);
            stats[i] += (tmp2[i]+min_time*(b+a*min_time)) * hazard[i];
        }
    }
    else { // update the statistics up to the final time
        for ( int i = 0; i < num_reactions; i++) {
            stats[i] += tmp2[i] * hazard[i];
        }
    }
    return(std::tuple<bool, double, int>(fired, delta_t, index));
}

// helper functions

vec PosteriorSimulator::get_control(vec& state, vec& backward) {
    // get target state list
    unsigned ind = model->state2ind(state);
    std::vector<int> targets = model->targets(state);
    // compute control factor
    return(get_control(ind, targets, backward));
}

vec PosteriorSimulator::get_control(int ind, std::vector<int>& targets, vec& backward) {
    // compute control factor
    vec control(targets.size());
    for (int i = 0; i < control.rows(); i++) {
        if ( targets[i] >= 0 ) {
            control[i] = backward[targets[i]] / backward[ind];
        } else {
            control[i] = 0.0;
        }
        // cut of very large controls
        if (control[i] > 1e10) {
            control[i] = 1e10;
        }
    }
    return(control);
}