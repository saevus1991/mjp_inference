#include "batched_filter.h"


pybind11::tuple batched_filter(np_array_c initial_in, np_array_c rates_in, MEInference* master_equation, ObservationModel* obs_model, np_array_c obs_times_in, np_array_c observations_in, np_array_c obs_param_in, bool get_gradient, int num_workers, std::string backend) {
    //set openmp 
    #ifdef _OPENMP
        int max_threads = omp_get_num_procs();
        if (num_workers > 0 && num_workers <= max_threads)
            omp_set_num_threads(num_workers);
    #endif 
    // get transition model
    MJP* transition_model = master_equation->get_model();
    // input checking
    unsigned num_states = transition_model->get_num_states();
    unsigned num_obs = obs_times_in.size();
    unsigned obs_dim = obs_model->get_obs_dim();
    unsigned obs_size = num_obs*obs_dim;
    unsigned num_obs_param = obs_model->get_num_param();
    unsigned num_rates = transition_model->get_num_rates();
    if (observations_in.shape(observations_in.ndim()-2) != num_obs) {
        std::string msg = "Number of observations must match number of observation times";
        throw std::invalid_argument(msg);        
    }
    if (initial_in.shape(initial_in.ndim()-1) != num_states) {
        std::string msg = "Size of array initial must match transition_model.get_num_states()";
        throw std::invalid_argument(msg);  
    }
    if (rates_in.shape(rates_in.ndim()-1) != num_rates) {
        std::string msg = "Size of array rates must match transition_model.get_num_param()";
        throw std::invalid_argument(msg);  
    }
    if (obs_param_in.shape(obs_param_in.ndim()-1) != num_obs_param) {
        std::string msg = "Size of array obs_param must match observation_model.get_num_param()";
        throw std::invalid_argument(msg);  
    }
    // get batchsize
    std::vector<pybind11::buffer_info> buffers;
    buffers.push_back(initial_in.request());
    buffers.push_back(rates_in.request());
    buffers.push_back(obs_param_in.request());
    buffers.push_back(observations_in.request());
    std::vector<int> base_dims({1, 1, 1, 2});
    int batch_size = ut::misc::infer_batchsize(buffers, base_dims);
    // parse input
    Eigen::Map<vec> initial((double*)initial_in.data(), initial_in.size());
    Eigen::Map<vec> rates((double*)rates_in.data(), rates_in.size());
    Eigen::Map<vec> obs_times((double*)obs_times_in.data(), obs_times_in.size());
    Eigen::Map<vec> observations((double*)observations_in.data(),observations_in.size());
    Eigen::Map<vec> obs_param((double*)obs_param_in.data(), obs_param_in.size());
    // parse getters
    std::function<vec (Eigen::Map<vec>&, int)> get_initial;
    if (initial_in.ndim() == base_dims[0]) {
        get_initial = [] (Eigen::Map<vec>& x, int i) { return( x ); };
    } else if (initial_in.ndim() == base_dims[0]+1 && initial_in.shape(0) == 1) {
        get_initial = [] (Eigen::Map<vec>& x, int i) { return( x ); };
    } else if (initial_in.ndim() == base_dims[0]+1 && initial_in.shape(0) == batch_size) {
        get_initial = [num_states] (Eigen::Map<vec>& x, int i) { return( x.segment(i*num_states, num_states) ); };
    } else {
        std::string msg = "Array initial must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    std::function<vec (Eigen::Map<vec>&, int)> get_rates;
    if (rates_in.ndim() == base_dims[1]) {
        get_rates = [] (Eigen::Map<vec>& x, int i) { return( x ); };
    } else if (rates_in.ndim() == base_dims[1]+1 && rates_in.shape(0) == 1) {
        get_rates = [] (Eigen::Map<vec>& x, int i) { return( x ); };
    } else if (rates_in.ndim() == base_dims[1]+1 && rates_in.shape(0) == batch_size) {
        get_rates = [num_rates] (Eigen::Map<vec>& x, int i) { return( x.segment(i*num_rates, num_rates) ); };
    } else {
        std::string msg = "Array rates must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    std::function<vec (Eigen::Map<vec>&, int)> get_obs_param;
    if (obs_param_in.ndim() == base_dims[2]) {
        get_obs_param = [] (Eigen::Map<vec>& x, int i) { return( x ); };
    } else if (obs_param_in.ndim() == base_dims[2]+1 && obs_param_in.shape(0) == 1) {
        get_obs_param = [] (Eigen::Map<vec>& x, int i) { return( x ); };
    } else if (obs_param_in.ndim() == base_dims[2]+1 && obs_param_in.shape(0) == batch_size) {
        get_obs_param = [num_obs_param] (Eigen::Map<vec>& x, int i) { return( x.segment(i*num_obs_param, num_obs_param) ); };
    } else {
        std::string msg = "Array obs_param must be one or two-dimensional";
        throw std::invalid_argument(msg);  
    }
    std::function<mat_rm (Eigen::Map<vec>&, int)> get_obs;
    if (observations_in.ndim() == base_dims[3]) {
        get_obs = [num_obs, obs_dim] (Eigen::Map<vec>& x, int i) { return( x.reshaped(num_obs, obs_dim) ); };
    } else if (observations_in.ndim() == base_dims[3]+1 && observations_in.shape(0) == 1) {
        get_obs = [num_obs, obs_dim] (Eigen::Map<vec>& x, int i) { return( x.reshaped(num_obs, obs_dim) ); };
    } else if (observations_in.ndim() == base_dims[3]+1 && observations_in.shape(0) == batch_size) {
        get_obs = [num_obs, obs_dim, obs_size] (Eigen::Map<vec>& x, int i) { return( x.segment(i*obs_size, obs_size).reshaped(num_obs, obs_dim) ); };
    } else {
        std::string msg = "Array observations must be two or three-dimensional";
        throw std::invalid_argument(msg);  
    }
    // set up output
    np_array res_out = np_array(batch_size);
    Eigen::Map<vec> res((double*)res_out.data(), res_out.size());
    np_array initial_grad_out(std::vector<int>({batch_size, int(num_states)}));
    Eigen::Map<vec> initial_grad((double*)initial_grad_out.data(), initial_grad_out.size());
    np_array rates_grad_out(std::vector<int>({batch_size, int(num_rates)}));
    Eigen::Map<vec> rates_grad((double*)rates_grad_out.data(), rates_grad_out.size());
    np_array obs_param_grad_out(std::vector<int>({batch_size, int(num_obs_param)}));
    Eigen::Map<vec> obs_param_grad((double*)obs_param_grad_out.data(), obs_param_grad_out.size());
    // perform computations
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        // choose backend
        if (backend == "krylov") {
            // compute likelihood
            KrylovFilter filt(master_equation, obs_model, obs_times, get_obs(observations, i), get_initial(initial, i), get_rates(rates, i), get_obs_param(obs_param, i));
            res[i] = filt.log_prob();
            if (get_gradient) {
                filt.log_prob_backward();
                filt.compute_rates_grad();
                initial_grad.segment(i*num_states, num_states) = filt.get_initial_grad();
                rates_grad.segment(i*num_rates, num_rates) = filt.get_rates_grad();
                obs_param_grad.segment(i*num_obs_param, num_obs_param) = filt.get_obs_param_grad();
            }
        } else if (backend == "krylov_mem") { // #TODO: unify this by pointer use
            // compute likelihood
            KrylovFilterMem filt(master_equation, obs_model, obs_times, get_obs(observations, i), get_initial(initial, i), get_rates(rates, i), get_obs_param(obs_param, i));
            res[i] = filt.log_prob();
            if (get_gradient) {
                filt.log_prob_backward();
                filt.compute_rates_grad();
                initial_grad.segment(i*num_states, num_states) = filt.get_initial_grad();
                rates_grad.segment(i*num_rates, num_rates) = filt.get_rates_grad();
                obs_param_grad.segment(i*num_obs_param, num_obs_param) = filt.get_obs_param_grad();
            }
        } else {
            std::invalid_argument exception("Unsupported backend " + backend);
            throw exception;
        }
    }
    if (get_gradient)
        return(pybind11::make_tuple(res_out, initial_grad_out, rates_grad_out, obs_param_grad_out));
    else
        return(pybind11::make_tuple(res_out));
}

pybind11::tuple batched_filter_list(np_array_c initial_in, np_array_c rates_in, MEInference* master_equation, ObservationModel* obs_model, pybind11::list obs_times_in, pybind11::list observations_in, np_array_c obs_param_in, bool get_gradient, int num_workers, std::string backend) {
    //set openmp 
    #ifdef _OPENMP
        int max_threads = omp_get_num_procs();
        if (num_workers > 0 && num_workers <= max_threads)
            omp_set_num_threads(num_workers);
    #endif
    // get transition model
    MJP* transition_model = master_equation->get_model();
    // input checking
    unsigned num_states = transition_model->get_num_states();
    unsigned num_obs_param = obs_model->get_num_param();
    unsigned obs_dim = obs_model->get_obs_dim();
    unsigned num_rates = transition_model->get_num_rates();
    if (initial_in.shape(initial_in.ndim()-1) != num_states) {
        std::string msg = "Size of array initial must match transition_model.get_num_states()";
        throw std::invalid_argument(msg);  
    }
    if (rates_in.shape(rates_in.ndim()-1) != num_rates) {
        std::string msg = "Size of array rates must match transition_model.get_num_param()";
        throw std::invalid_argument(msg);  
    }
    if (obs_param_in.shape(obs_param_in.ndim()-1) != num_obs_param) {
        std::string msg = "Size of array obs_param must match observation_model.get_num_param()";
        throw std::invalid_argument(msg);  
    }
    // get batchsize
    std::vector<pybind11::buffer_info> buffers;
    buffers.push_back(initial_in.request());
    buffers.push_back(rates_in.request());
    buffers.push_back(obs_param_in.request());
    std::vector<int> base_dims({1, 1, 1});
    std::vector<int> tmp = {ut::misc::infer_batchsize(buffers, base_dims), int(obs_times_in.size()), int(observations_in.size())};
    int batch_size = *std::max_element(tmp.begin(), tmp.end());
    // parse input  #TODO: use map_array in other batched versions
    std::vector<Eigen::Map<vec>> initial = ut::misc::map_array(initial_in, batch_size, 1);
    std::vector<Eigen::Map<vec>> rates = ut::misc::map_array(rates_in, batch_size, 1);
    std::vector<Eigen::Map<vec>> obs_param = ut::misc::map_array(obs_param_in, batch_size, 1);
    std::vector<Eigen::Map<vec>> obs_times = ut::misc::parse_array_list(obs_times_in, batch_size);
    std::vector<Eigen::Map<vec>> observations = ut::misc::parse_array_list(observations_in, batch_size);
    // set up output
    np_array res_out = np_array(batch_size);
    Eigen::Map<vec> res((double*)res_out.data(), res_out.size());
    np_array initial_grad_out(std::vector<int>({batch_size, int(num_states)}));
    Eigen::Map<vec> initial_grad((double*)initial_grad_out.data(), initial_grad_out.size());
    np_array rates_grad_out(std::vector<int>({batch_size, int(num_rates)}));
    Eigen::Map<vec> rates_grad((double*)rates_grad_out.data(), rates_grad_out.size());
    np_array obs_param_grad_out(std::vector<int>({batch_size, int(num_obs_param)}));
    Eigen::Map<vec> obs_param_grad((double*)obs_param_grad_out.data(), obs_param_grad_out.size());
    // perform computations
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        // choose backend
        if (backend == "krylov") {
            // compute likelihood
            unsigned num_obs = obs_times[i].size();
            KrylovFilter filt = KrylovFilter(master_equation, obs_model, obs_times[i], observations[i].reshaped(num_obs, obs_dim), initial[i], rates[i], obs_param[i]);
            res[i] = filt.log_prob();
            if (get_gradient) {
                filt.log_prob();
                filt.compute_rates_grad();
                initial_grad.segment(i*num_states, num_states) = filt.get_initial_grad();
                rates_grad.segment(i*num_rates, num_rates) = filt.get_rates_grad();
                obs_param_grad.segment(i*num_obs_param, num_obs_param) = filt.get_obs_param_grad();
            }
        } else {
            std::invalid_argument exception("Unsupported backend " + backend);
            throw exception;
        }
    }
    if (get_gradient)
        return(pybind11::make_tuple(res_out, initial_grad_out, rates_grad_out, obs_param_grad_out));
    else
        return(pybind11::make_tuple(res_out));
}
