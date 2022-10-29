#include "obs_model.h"

// constructor

ObservationModel::ObservationModel(MJP* transition_model_, const std::string& noise_type_) :
    transition_model(transition_model_),
    noise_type(noise_type_),
    noise_model(build_noise_model()),
    noise_param_list(noise_model->get_param_list()),
    num_param(0),
    obs_dim(1)
    {}

ObservationModel::~ObservationModel() {
    delete noise_model;
}

// helpers

NoiseModel* ObservationModel::build_noise_model() {
    NoiseModel* noise_model;
    if (noise_type == "normal") {
        noise_model = new Normal();
    } else {
        std::string msg = "Noise type " + noise_type + " not yet supported";
        throw std::invalid_argument(msg);
    }
    return(noise_model);
}

void ObservationModel::build() {
    // get list of transforms
    std::vector<std::string> transform_list;
    for (unsigned i = 0; i < transform_map.size(); i++) {
        transform_list.push_back(transform_map[i].get_name());
    }
    // create new list matching the noise params
    std::vector<Transform> transform_map_new;
    for (unsigned i = 0; i < noise_param_list.size(); i++) {
        auto it = std::find(transform_list.begin(), transform_list.end(), noise_param_list[i]);
        if (it != transform_list.end()) {
            unsigned ind = it - transform_list.begin();
            transform_map_new.push_back(transform_map[ind]);
        } else {
            std::string msg = "No transform provided for noise model parameter " + noise_param_list[i];
            throw std::invalid_argument(msg);
        }
    }
    // update transform map
    transform_map = transform_map_new;
    // make param vec
    param_array = vec(num_param);
    for (unsigned i = 0; i < num_param; i++) {
        param_array[i] = param_map[i].get_value()[0]; // only for 1-d params
    }
    // infer obs dimension
    vec state = ut::vec2vec(transition_model->get_default_state());
    vec obs = sample_np(0.0, state, param_array, 0);
    obs_dim = obs.size();
}

// getters

std::string ObservationModel::get_param_parser() const {
    std::string param_parser;
    for (unsigned i = 0; i < param_list.size(); i++) {
        std::string line = param_list[i] + " = param[" + std::to_string(i) + "] \n";
        param_parser += line;
    };
    return(param_parser);
}