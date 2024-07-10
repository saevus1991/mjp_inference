#include "simulate.h"


mat_rm simulate(MJP* transition_model, ObservationModel* obs_model, const Eigen::Map<vec>& initial, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& obs_param, const Eigen::Map<vec>& t_eval, std::mt19937* rng, int max_events, std::string max_event_handler) {
    // simulate states
    vec tspan = t_eval(std::vector<int>{0, int(t_eval.rows())-1});
    Simulator simulator(transition_model, initial, rates, tspan, rng, max_events, max_event_handler);
    mat_rm states = simulator.simulate(t_eval);
    // create observations
    if (obs_model != nullptr) {
        // preparations
        unsigned num_steps = t_eval.size();
        unsigned obs_dim = obs_model->get_obs_dim();
        vec obs_param_ = obs_param; 
        // create output
        mat_rm obs(num_steps, obs_dim);
        for (unsigned i = 0; i < num_steps; i++) {
            vec state_i = states.row(i).transpose();
            obs.row(i).noalias() = obs_model->sample(t_eval[i], state_i, obs_param_, rng);
        }
        return(obs);
    } else {
        return(states);
    }
}

mat_rm simulate(MJP* transition_model, ObservationModel* obs_model, const Eigen::Map<vec>& initial, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& obs_param, const Eigen::Map<vec>& t_eval, int seed, int max_events, std::string max_event_handler) {
    // set up rng
    std::mt19937 rng(seed);
    return(simulate(transition_model, obs_model, initial, rates, obs_param, t_eval, &rng, max_events, max_event_handler));
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
    if (initial_dist_in.ndim() > 2) {
        std::string msg = "Array initial_dist must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    if (rates_in.ndim() > 2) {
        std::string msg = "Array rates must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    if (obs_param_in.ndim() > 2) {
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
    std::vector<Eigen::Map<vec>> initial_dist = ut::misc::map_array(initial_dist_in, batch_size, 1);
    std::vector<Eigen::Map<vec>> rates = ut::misc::map_array(rates_in, batch_size, 1);
    std::vector<Eigen::Map<vec>> obs_param = ut::misc::map_array(obs_param_in, batch_size, 1);
    Eigen::Map<vec> obs_times((double*)obs_times_in.data(), obs_times_in.size());
    // get relevant sizes
    int num_states = transition_model->get_num_states();
    int num_obs = obs_times.rows();
    int obs_dim = obs_model->get_obs_dim();
    int num_obs_param = obs_model->get_num_param();
    int num_rates = transition_model->get_num_rates();
    // set up output
    std::vector<int> output_shape({batch_size, num_obs, obs_dim});
    np_array output_(output_shape);
    std::vector<Eigen::Map<mat_rm>> output;
    for (unsigned i = 0; i < batch_size; i++) {
        auto tmp = output_[pybind11::slice(i, i+1, 1)].cast<np_array_c>();
        double* output_ptr_i = (double*) tmp.data();
        output.push_back(Eigen::Map<mat_rm>(output_ptr_i, num_obs, obs_dim));
    }
    // perform computations
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        // set up rng
        std::mt19937 rng(seed+i);
        // simulate initial 
        std::vector<double> initial_dist_std(initial_dist[i].data(), initial_dist[i].data()+num_states);
        std::discrete_distribution<int> dist(initial_dist_std.begin(), initial_dist_std.end());
        vec initial_i = transition_model->ind2state(dist(rng));
        Eigen::Map<vec> initial_i_map(initial_i.data(), initial_i.size()); // required for copatibility with simulate
        // simulate trajectory
        output[i].noalias() = simulate(transition_model, obs_model, initial_i_map, rates[i], obs_param[i], obs_times, seed+i, max_events, max_event_handler);
    }
    return(output_);
}

Trajectory simulate_posterior(MJP* transition_model, const Eigen::Map<vec>& initial, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& tspan, const Eigen::Map<vec>& t_grid, const Eigen::Map<mat_rm>& backward_grid, std::mt19937* rng, int max_events, std::string max_event_handler) {
    // set up simulator
    PosteriorSimulator simulator(transition_model, initial, tspan, rng, max_events, max_event_handler);
    // simulate 
    return(simulator.simulate(t_grid, backward_grid));
}

Trajectory simulate_posterior(MEInference* master_equation, ObservationModel* obs_model, const Eigen::Map<vec>& initial_dist, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& obs_param, const Eigen::Map<vec>& tspan, const Eigen::Map<vec>& obs_times, const Eigen::Map<mat_rm>& observations, const Eigen::Map<vec>& t_grid, std::mt19937* rng, int max_events, std::string max_event_handler) {
    // compute backward solution
    KrylovBackwardFilter filt = KrylovBackwardFilter(master_equation, obs_model, obs_times, observations, initial_dist, rates, obs_param, tspan);
    filt.backward_filter();
    mat_rm backward_grid = filt.eval_backward_filter(t_grid);
    ut::math::project_positive(backward_grid);
    // set up smoothing for initial dist
    std::vector<double> initial_smoothed_std(initial_dist.rows());
    Eigen::Map<vec> initial_smoothed(initial_smoothed_std.data(), initial_smoothed_std.size());
    initial_smoothed.noalias() = filt.get_smoothed_initial();
    // simulate initial
    MJP* transition_model = master_equation->get_model();
    vec initial(transition_model->get_num_species());
    std::discrete_distribution<int> dist(initial_smoothed_std.begin(), initial_smoothed_std.end());
    initial = transition_model->ind2state(dist(*rng));
    // create trajectory
    Eigen::Map<vec> initial_map(initial.data(), initial.size());
    Eigen::Map<mat_rm> backward_grid_map(backward_grid.data(), backward_grid.rows(), backward_grid.cols());
    return(simulate_posterior(transition_model, initial_map, rates, tspan, t_grid, backward_grid_map, rng, max_events, max_event_handler));
}

Trajectory simulate_posterior(MEInference* master_equation, ObservationModel* obs_model, const Eigen::Map<vec>& initial_dist, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& obs_param, const Eigen::Map<vec>& tspan, const Eigen::Map<vec>& obs_times, const Eigen::Map<mat_rm>& observations, const Eigen::Map<vec>& t_grid, int seed, int max_events, std::string max_event_handler) {
    // get generator
    std::mt19937 rng(seed);
    // simulate
    return(simulate_posterior(master_equation, obs_model, initial_dist, rates, obs_param, tspan, obs_times, observations, t_grid, &rng, max_events, max_event_handler));
}

pybind11::dict simulate_posterior(MEInference* master_equation, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c rates_in, np_array_c obs_param_in, np_array_c tspan_in, np_array_c obs_times_in, np_array_c observations_in, np_array_c t_grid_in, int seed,  int max_events, std::string max_event_handler) {
    // parse input
    Eigen::Map<vec> initial_dist((double*) initial_dist_in.data(), initial_dist_in.size());
    Eigen::Map<vec> rates((double*) rates_in.data(), rates_in.size());
    Eigen::Map<vec> obs_param((double*) obs_param_in.data(), obs_param_in.size());
    Eigen::Map<vec> tspan((double*) tspan_in.data(), tspan_in.size());
    Eigen::Map<vec> obs_times((double*) obs_times_in.data(), obs_times_in.size());
    Eigen::Map<vec> t_grid((double*) t_grid_in.data(), t_grid_in.size());
    Eigen::Map<mat_rm> observations((double*) observations_in.data(), observations_in.shape(0), observations_in.shape(1));
    // simulate
    return(simulate_posterior(master_equation, obs_model, initial_dist, rates, obs_param, tspan, obs_times, observations, t_grid, seed, max_events, max_event_handler).to_dict());
}


pybind11::list simulate_posterior_batched(MEInference* master_equation, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c rates_in, np_array_c obs_param_in,np_array_c tspan_in, np_array_c obs_times_in, np_array_c observations_in, np_array_c t_grid_in, int seed, int num_samples, int num_workers, int max_events, std::string max_event_handler) {
    // #TODO: This failed with the poisson example -> check
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
    // get transition model
    MJP* transition_model = master_equation->get_model();
    // infer batch size
    std::vector<pybind11::buffer_info> buffers;
    buffers.push_back(initial_dist_in.request());
    buffers.push_back(rates_in.request());
    buffers.push_back(obs_param_in.request());
    buffers.push_back(observations_in.request());
    std::vector<int> base_dims({1, 1, 1, 2});
    int batch_size = ut::misc::infer_batchsize(buffers, base_dims, num_samples);
    // input checking
    if (initial_dist_in.ndim() > 2) {
        std::string msg = "Array initial_dist must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    if (rates_in.ndim() > 2) {
        std::string msg = "Array rates must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    if (obs_param_in.ndim() > 2) {
        std::string msg = "Array obs_param must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    if (observations_in.ndim() > 3) {
        std::string msg = "Array observations must be two or three-dimensional";
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
    if (observations_in.shape(observations_in.ndim()-1) != obs_model->get_obs_dim()) {
        std::string msg = "Shape of array observations does not match dimensions expected from obs_model";
        throw std::invalid_argument(msg);  
    }
    if (observations_in.shape(observations_in.ndim()-2) != obs_times_in.size()) {
        std::string msg = "Shape of array observations does not match size expected from obs_times";
        throw std::invalid_argument(msg);  
    }
    // parse input
    std::vector<Eigen::Map<vec>> initial_dist = ut::misc::map_array(initial_dist_in, batch_size, 1);
    std::vector<Eigen::Map<vec>> rates = ut::misc::map_array(rates_in, batch_size, 1);
    std::vector<Eigen::Map<vec>> obs_param = ut::misc::map_array(obs_param_in, batch_size, 1);
    std::vector<Eigen::Map<vec>> observations = ut::misc::map_array(obs_param_in, batch_size, 1);
    Eigen::Map<vec> obs_times((double*)obs_times_in.data(), obs_times_in.size());
    Eigen::Map<vec> t_grid((double*)t_grid_in.data(), t_grid_in.size());
    Eigen::Map<vec> tspan((double*)tspan_in.data(), tspan_in.size());
    // get relevant sizes
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
    bool identical_backward = obs_param_in.size() == num_obs_param && observations_in.size() == num_obs*obs_dim && initial_dist_in.size() == transition_model->get_num_states() && rates_in.size() == num_rates;
    // perform computations
    if (identical_backward) {
        // set up backward filter
        Eigen::Map<mat_rm> observations_(observations[0].data(), num_obs, obs_dim);
        KrylovBackwardFilter filt = KrylovBackwardFilter(master_equation, obs_model, obs_times, observations_, initial_dist[0], rates[0], obs_param[0], tspan);
        filt.backward_filter();
        mat_rm backward_grid = filt.eval_backward_filter(t_grid);
        ut::math::project_positive(backward_grid);
        Eigen::Map<mat_rm> backward_grid_map(backward_grid.data(), backward_grid.rows(), backward_grid.cols());
        // set up smoothing for initial dist
        std::vector<double> initial_smoothed_std(initial_dist[0].rows());
        Eigen::Map<vec> initial_smoothed(initial_smoothed_std.data(), initial_smoothed_std.size());
        initial_smoothed.noalias() = filt.get_smoothed_initial();
        // iterate over batch
        #pragma omp parallel for
        for (int i = 0; i < batch_size; i++) {
            // get generator
            std::mt19937 rng(seed+i);
            // simulate initial
            std::discrete_distribution<int> dist(initial_smoothed_std.begin(), initial_smoothed_std.end());
            vec initial = transition_model->ind2state(dist(rng));
            // simulate trajectory
            Eigen::Map<vec> initial_map(initial.data(), initial.size());
            trajectories[i] = simulate_posterior(transition_model, initial_map, rates[i], tspan, t_grid, backward_grid_map, &rng, max_events, max_event_handler);
        }
    } else {
        #pragma omp parallel for
        for (int i = 0; i < batch_size; i++) {
            // get generator
            std::mt19937 rng(seed+i);
            // simulate
            Eigen::Map<mat_rm> observations_i(observations[i].data(), num_obs, obs_dim);
            trajectories[i] = simulate_posterior(master_equation, obs_model, initial_dist[i], rates[i], obs_param[i], tspan, obs_times, observations_i, t_grid, &rng, max_events, max_event_handler);
        }
    }
    // convert output to python
    pybind11::list trajectories_out;
    for (int i = 0; i < batch_size; i++) {
        trajectories_out.append(trajectories[i].to_dict());
        trajectories[i].clear(); 
    }
    return(trajectories_out);
}