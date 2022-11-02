#include "simulate.h"


mat_rm simulate(MJP* transition_model, ObservationModel* obs_model, const Eigen::Map<vec>& initial, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& obs_param, const Eigen::Map<vec>& t_eval, int seed, int max_events, std::string max_event_handler) {
    // set up rng
    std::mt19937 rng(seed);
    // simulate states
    vec tspan = t_eval(std::vector<int>{0, int(t_eval.rows())-1});
    Simulator simulator(transition_model, initial, rates, tspan, &rng, max_events, max_event_handler);
    mat_rm states = simulator.simulate(t_eval);
    // create observations
    if (obs_model != nullptr) {
        // preparations
        unsigned num_steps = t_eval.size();
        unsigned obs_dim = obs_model->get_obs_dim();
        vec obs_param_ = obs_param; // #TODO: use const double in obs->sample, then this extra step can be removed
        // create output
        mat_rm obs(num_steps, obs_dim);
        for (unsigned i = 0; i < num_steps; i++) {
            vec state_i = states.row(i).transpose();
            obs.row(i).noalias() = obs_model->sample(t_eval[i], state_i, obs_param_, &rng);
        }
        return(obs);
    } else {
        return(states);
    }
}

np_array simulate(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_in, np_array_c rates_in, np_array_c obs_param_in,  np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler) {
    // parse input
    Eigen::Map<vec> initial((double*) initial_in.data(), initial_in.size());
    Eigen::Map<vec> rates((double*) rates_in.data(), rates_in.size());
    Eigen::Map<vec> obs_param((double*) obs_param_in.data(), obs_param_in.size());
    Eigen::Map<vec> t_eval((double*) t_eval_in.data(), t_eval_in.size());
    // simulate
    return(ut::mat2array(simulate(transition_model, obs_model, initial, rates, obs_param, t_eval, seed, max_events, max_event_handler)));
}

np_array simulate(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_in, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler) {
    // get defaults
    np_array rates = transition_model->get_rate_array_np();
    np_array obs_param = obs_model->get_param_array_np();
    // simulate
    return(simulate(transition_model, obs_model, initial_in, rates, obs_param, t_eval_in, seed, max_events, max_event_handler));
}

np_array simulate(MJP* transition_model, np_array_c initial_in, np_array_c rates_in, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler) {
    // get obs_model
    ObservationModel* obs_model = nullptr;
    np_array_c obs_param;
    // simulate
    return(simulate(transition_model, obs_model, initial_in, rates_in, obs_param, t_eval_in, seed, max_events, max_event_handler));
}

np_array simulate(MJP* transition_model, np_array_c initial_in, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler) {
    // get rates
    np_array rates = transition_model->get_rate_array_np();
    // simulate
    return(simulate(transition_model, initial_in, rates, t_eval_in, seed, max_events, max_event_handler));
}

pybind11::dict simulate_full(MJP* transition_model, np_array_c initial_, np_array_c rates_, np_array_c tspan_, int seed, int max_events, std::string max_event_handler) {
    // parse input
    Eigen::Map<vec> initial((double*) initial_.data(), initial_.size());
    Eigen::Map<vec> rates((double*) rates_.data(), rates_.size());
    Eigen::Map<vec> tspan((double*) tspan_.data(), tspan_.size());
    // simulate
    return(simulate_full(transition_model, initial, rates, tspan, seed, max_events, max_event_handler).to_dict());
}

pybind11::dict simulate_full(MJP* transition_model, np_array_c initial_, np_array_c tspan_, int seed, int max_events, std::string max_event_handler) {
    // get rates
    np_array rates = transition_model->get_rate_array_np();
    // simulate
    return(simulate_full(transition_model, initial_, rates, tspan_, seed, max_events, max_event_handler));
}

np_array simulate_batched(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c rates_in, np_array_c obs_param_in, np_array_c obs_times_in, int seed, int num_samples, int num_workers, int max_events, std::string max_event_handler) {
    //set openmp 
    #ifdef _OPENMP
        int max_threads = omp_get_num_procs();
        if (num_workers > 0 && num_workers <= max_threads) {
            omp_set_num_threads(num_workers);
        } else if (num_workers == -1) {
            omp_set_num_threads(max_threads);
        } else {
            std::string msg = "Invalid number of workers " + std::to_string(num_workers);
            throw std::invalid_argument(msg);
        } 
    #endif
    // infer batch size
    std::vector<pybind11::buffer_info> buffers;
    buffers.push_back(initial_dist_in.request());
    buffers.push_back(rates_in.request());
    buffers.push_back(obs_param_in.request());
    std::vector<int> base_dims({1, 1, 1, 1});
    int batch_size = ut::misc::infer_batchsize(buffers, base_dims, num_samples);
    // input checking
    std::function<vec (Eigen::Map<vec>&, int, int)> get_initial;
    if (initial_dist_in.ndim() == 1) {
        get_initial = [] (Eigen::Map<vec>& x, int i, int dim) { return( x ); };
    } else if (initial_dist_in.ndim() == 2 && initial_dist_in.shape(0) == batch_size) {
        get_initial = [] (Eigen::Map<vec>& x, int i, int dim) { return( x.segment(i*dim, dim) ); };
    } else {
        std::string msg = "Array initial_dist must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    std::function<vec (Eigen::Map<vec>&, int, int)> get_rates;
    if (rates_in.ndim() == 1) {
        get_rates = [] (Eigen::Map<vec>& x, int i, int dim) { return( x ); };
    } else if (rates_in.ndim() == 2 && rates_in.shape(0) == batch_size) {
        get_rates = [] (Eigen::Map<vec>& x, int i, int dim) { return( x.segment(i*dim, dim) ); };
    } else {
        std::string msg = "Array rates must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    std::function<vec (Eigen::Map<vec>&, int, int)> get_obs_param;
    if (obs_param_in.ndim() == 1) {
        get_obs_param = [] (Eigen::Map<vec>& x, int i, int dim) { return( x ); };
    } else if (obs_param_in.ndim() == 2 && obs_param_in.shape(0) == batch_size) {
        get_obs_param = [] (Eigen::Map<vec>& x, int i, int dim) { return( x.segment(i*dim, dim) ); };
    } else {
        std::string msg = "Array obs_param must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    if (initial_dist_in.shape(initial_dist_in.ndim()-1) != transition_model->get_num_states()) {
        std::string msg = "Size of array initial must match transition_model.get_num_states()";
        throw std::invalid_argument(msg);  
    }
    if (rates_in.shape(rates_in.ndim()-1) != transition_model->get_num_rates()) {
        std::string msg = "Size of array rates must match transition_model.get_num_param()";
        throw std::invalid_argument(msg);  
    }
    if (obs_param_in.shape(obs_param_in.ndim()-1) != obs_model->get_num_param()) {
        std::string msg = "Size of array obs_param must match observation_model.get_num_param()";
        throw std::invalid_argument(msg);  
    }
    // parse input
    Eigen::Map<vec> initial_dist((double*)initial_dist_in.data(), initial_dist_in.size());
    Eigen::Map<vec> rates((double*)rates_in.data(), rates_in.size());
    Eigen::Map<vec> obs_times((double*)obs_times_in.data(), obs_times_in.size());
    Eigen::Map<vec> obs_param((double*)obs_param_in.data(), obs_param_in.size());
    vec tspan = obs_times(std::vector<int>{0, int(obs_times.rows())-1});
    int num_states = transition_model->get_num_states();
    int num_obs = obs_times.rows();
    int obs_dim = obs_model->get_obs_dim();
    int num_obs_param = obs_model->get_num_param();
    int num_rates = transition_model->get_num_rates();
    // set up output
    std::map<int, mat_rm> obs;
    for (int i = 0; i < batch_size; i++) {
        obs[i] = mat_rm();
    }
    // perform computations
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        // set up rng
        std::mt19937 rng(seed+i);
        // create local copies
        vec rates_i = get_rates(rates, i, num_rates);
        vec obs_param_i = get_obs_param(obs_param, i, num_obs_param);
        // set up simulator
        Simulator simulator(transition_model, transition_model->get_num_species(), rates_i, tspan, &rng, max_events, max_event_handler);
        // draw initial
        std::vector<double> initial_dist_std(num_states);
        Eigen::Map<vec> initial_dist_i(initial_dist_std.data(), initial_dist_std.size());
        initial_dist_i.noalias() = get_initial(initial_dist, i, num_states);
        std::discrete_distribution<int> dist(initial_dist_std.begin(), initial_dist_std.end());
        vec initial_i = transition_model->ind2state(dist(rng));
        simulator.set_initial(initial_i);
        // simulate states
        mat_rm states = simulator.simulate(obs_times);
        // generate observations
        obs[i] = mat_rm(num_obs, obs_dim);
        for (int j = 0; j < num_obs; j++) {
            vec state_j = states.row(j).transpose();
            vec obs_j = obs_model->sample(obs_times[j], state_j, obs_param_i, &rng);
            obs[i].row(j) = obs_j.transpose();
        }
    }
    // convert output to array
    np_array obs_out(std::vector<int>({batch_size, num_obs, obs_dim}));
    for (int i = 0; i < batch_size; i++) {
        double* obs_out_ptr = (double*) obs_out.data() + i*num_obs*obs_dim;
        double* obs_i_ptr = (double*) obs[i].data();
        ut::copy_buffer(obs_i_ptr, obs_out_ptr, num_obs*obs_dim);
    }
    return(obs_out);
}

// np_array simulate(np_array_c initial_in, TransitionModel& transition_model, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler) {
//     // parse input
//     Eigen::Map<vec> initial((double*) initial_in.data(), initial_in.size());
//     Eigen::Map<vec> t_eval((double*) t_eval_in.data(), t_eval_in.size());
//     // simulate states
//     vec tspan = t_eval(std::vector<int>{0, int(t_eval.rows())-1});
//     Simulator simulator(transition_model, initial, tspan, seed, max_events, max_event_handler);
//     mat_rm states = simulator.simulate(t_eval);
//     // create observations
//     int num_steps = t_eval_in.size();
//     int num_sites = transition_model.get_num_sites();
//     std::mt19937& rng = simulator.get_rng();
//     np_array obs_out(std::vector<int>({num_steps, num_sites}));
//     Eigen::Map<vec> obs((double*) obs_out.data(), obs_out.size());
//     for (int i = 0; i < num_steps; i++) {
//         vec state_i = states.row(i).transpose();
//         obs.segment(i*num_sites, num_sites) = state_i;
//     }
//     return(obs_out);
// }

pybind11::object simulate_posterior(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c t_obs_in, np_array_c observations_in, np_array_c tspan_in, np_array_c t_grid_in, int seed, int num_samples, int max_events, std::string max_event_handler) {
    // checks
    if (initial_dist_in.size() != transition_model->get_num_states()) {
        std::string msg = "Size of arguments initial_dist must match transition_model.get_num_states()";
        throw std::invalid_argument(msg); 
    }
    // parse input
    Eigen::Map<vec> initial_dist((double*) initial_dist_in.data(), initial_dist_in.size());
    Eigen::Map<vec> t_obs((double*) t_obs_in.data(), t_obs_in.size());
    Eigen::Map<vec> t_grid((double*) t_grid_in.data(), t_grid_in.size());
    vec tspan = Eigen::Map<vec> ((double*) tspan_in.data(), tspan_in.size());
    mat_rm observations = Eigen::Map<mat_rm> ((double*) observations_in.data(), observations_in.shape(0), observations_in.shape(1));
    vec rates = transition_model->get_rate_array();
    vec obs_param = obs_model->get_param_array();
    // compute backward solution
    MEInference master_equation(transition_model);
    KrylovBackwardFilter filt = KrylovBackwardFilter(&master_equation, obs_model, t_obs, observations, initial_dist, rates, obs_param, tspan);
    filt.backward_filter();
    mat_rm backward_grid = filt.eval_backward_filter(t_grid);
    // set up smoothing for initial dist
    std::vector<double> initial_smoothed_std(initial_dist.rows());
    Eigen::Map<vec> initial_smoothed(initial_smoothed_std.data(), initial_smoothed_std.size());
    initial_smoothed.noalias() = filt.get_smoothed_initial();
    // get generator
    std::mt19937 rng(seed);
    // simulate trajectory
    vec initial(transition_model->get_num_species());
    PosteriorSimulator simulator(transition_model, initial, tspan, &rng, max_events, max_event_handler);
    std::discrete_distribution<int> dist(initial_smoothed_std.begin(), initial_smoothed_std.end());
    if (num_samples == 1) {
        initial = transition_model->ind2state(dist(rng));
        simulator.set_initial(initial);
        pybind11::dict trajectory = simulator.simulate(t_grid, backward_grid).to_dict();
        return(trajectory);
    } else {
        pybind11::list trajectories;
        for (int i = 0; i < num_samples; i++) {
            initial = transition_model->ind2state(dist(rng));
            simulator.set_initial(initial);
            trajectories.append(simulator.simulate(t_grid, backward_grid).to_dict());
        }
        return(trajectories);
    }
}

pybind11::list simulate_posterior_batched(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c rates_in, np_array_c obs_times_in, np_array_c observations_in,np_array_c obs_param_in, np_array_c tspan_in, np_array_c t_grid_in, int seed, int num_workers, int max_events, std::string max_event_handler) {
    //set openmp 
    #ifdef _OPENMP
        int max_threads = omp_get_num_procs();
        if (num_workers > 0 && num_workers <= max_threads) {
            omp_set_num_threads(num_workers);
        } else if (num_workers == -1) {
            omp_set_num_threads(max_threads);
        } else {
            std::string msg = "Invalid number of workers " + std::to_string(num_workers);
            throw std::invalid_argument(msg);
        } 
    #endif
    // input checking
    if (observations_in.ndim() != 3) {
        std::string msg = "Array observations must be of dimension 3";
        throw std::invalid_argument(msg);        
    }
    int batch_size = observations_in.shape(0);
    std::function<vec (Eigen::Map<vec>&, int, int)> get_initial;
    if (initial_dist_in.ndim() == 1) {
        get_initial = [] (Eigen::Map<vec>& x, int i, int dim) { return( x ); };
    } else if (initial_dist_in.ndim() == 2 && initial_dist_in.shape(0) == batch_size) {
        get_initial = [] (Eigen::Map<vec>& x, int i, int dim) { return( x.segment(i*dim, dim) ); };
    } else {
        std::string msg = "Array initial must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    std::function<vec (Eigen::Map<vec>&, int, int)> get_rates;
    if (rates_in.ndim() == 1) {
        get_rates = [] (Eigen::Map<vec>& x, int i, int dim) { return( x ); };
    } else if (rates_in.ndim() == 2 && rates_in.shape(0) == batch_size) {
        get_rates = [] (Eigen::Map<vec>& x, int i, int dim) { return( x.segment(i*dim, dim) ); };
    } else {
        std::string msg = "Array rates must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    std::function<vec (Eigen::Map<vec>&, int, int)> get_obs_param;
    if (obs_param_in.ndim() == 1) {
        get_obs_param = [] (Eigen::Map<vec>& x, int i, int dim) { return( x ); };
    } else if (obs_param_in.ndim() == 2 && obs_param_in.shape(0) == batch_size) {
        get_obs_param = [] (Eigen::Map<vec>& x, int i, int dim) { return( x.segment(i*dim, dim) ); };
    } else {
        std::string msg = "Array obs_param must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    if (initial_dist_in.shape(initial_dist_in.ndim()-1) != transition_model->get_num_states()) {
        std::string msg = "Size of array initial must match transition_model.get_num_states()";
        throw std::invalid_argument(msg);  
    }
    if (rates_in.shape(rates_in.ndim()-1) != transition_model->get_num_rates()) {
        std::string msg = "Size of array rates must match transition_model.get_num_param()";
        throw std::invalid_argument(msg);  
    }
    if (obs_param_in.shape(obs_param_in.ndim()-1) != obs_model->get_num_param()) {
        std::string msg = "Size of array obs_param must match observation_model.get_num_param()";
        throw std::invalid_argument(msg);  
    }
    // parse input
    Eigen::Map<vec> initial_dist((double*)initial_dist_in.data(), initial_dist_in.size());
    Eigen::Map<vec> rates((double*)rates_in.data(), rates_in.size());
    Eigen::Map<vec> obs_times((double*)obs_times_in.data(), obs_times_in.size());
    Eigen::Map<vec> observations((double*)observations_in.data(),observations_in.size());
    Eigen::Map<vec> obs_param((double*)obs_param_in.data(), obs_param_in.size());
    Eigen::Map<vec> tspan((double*)tspan_in.data(), tspan_in.size());
    Eigen::Map<vec> t_grid((double*)t_grid_in.data(), t_grid_in.size());
    int num_states = transition_model->get_num_states();
    int num_obs = obs_times.rows();
    int obs_dim = obs_model->get_obs_dim();
    int num_obs_param = obs_model->get_num_param();
    int num_rates = transition_model->get_num_rates();
    // set up output
    std::map<int, Trajectory> trajectories;
    for (int i = 0; i < batch_size; i++) {
        trajectories[i] = Trajectory();
    }
    // setup me
    MEInference master_equation(transition_model); // #TODO: make simulate take master equation as argument
    // perform computations
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        // create local copies
        mat_rm observations_i = observations.segment(i*num_obs*obs_dim, num_obs*obs_dim).reshaped(num_obs, obs_dim);
        vec initial_dist_i =  get_initial(initial_dist, i, num_states);
        vec rates_i = get_rates(rates, i, num_rates);
        vec obs_param_i = get_obs_param(obs_param, i, num_obs_param);
        // set up backward filter
        KrylovBackwardFilter filt = KrylovBackwardFilter(&master_equation, obs_model, obs_times, observations_i, initial_dist_i, rates_i, obs_param_i, tspan);
        filt.backward_filter();
        mat_rm backward_grid = filt.eval_backward_filter(t_grid);
        ut::math::project_positive(backward_grid);
        // set up smoothing for initial dist
        std::vector<double> initial_smoothed_std(num_states);
        Eigen::Map<vec> initial_smoothed(initial_smoothed_std.data(), initial_smoothed_std.size());
        initial_smoothed.noalias() = filt.get_smoothed_initial();
        // get generator
        std::mt19937 rng(seed+i);
        // set up simulator
        PosteriorSimulator simulator(transition_model, transition_model->get_num_species(), rates_i, tspan, &rng, max_events, max_event_handler);
        // draw initial
        std::discrete_distribution<int> dist(initial_smoothed_std.begin(), initial_smoothed_std.end());
        vec initial_i = transition_model->ind2state(dist(rng));
        simulator.set_initial(initial_i);
        // simulate trajectory
        trajectories[i] = simulator.simulate(t_grid, backward_grid);
    }
    // convert output to python
    pybind11::list trajectories_out;
    for (int i = 0; i < batch_size; i++) {
        trajectories_out.append(trajectories[i].to_dict());
        trajectories[i].clear();
    }
    return(trajectories_out);
}