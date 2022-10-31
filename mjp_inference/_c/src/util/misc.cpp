#include "misc.h"

int ut::misc::infer_batchsize(std::vector<pybind11::buffer_info>& buffers, const std::vector<int>& base_dims, int batchsize_) {
    int num_arrays = buffers.size();
    // get all batch dimensions
    std::vector<int> batch_size_candidates;
    batch_size_candidates.push_back(1);
    for (int i = 0; i < num_arrays; i++) {
        if (buffers[i].ndim > base_dims[i]) {
            batch_size_candidates.push_back(buffers[i].shape[0]);
        }
    }
    // extract maximum
    int batch_size = *std::max_element(batch_size_candidates.begin(), batch_size_candidates.end());
    // check for consistency
    for (int i = 0; i < batch_size_candidates.size(); i++) {
        if ( !( batch_size_candidates[i] == 1 || batch_size_candidates[i] == batch_size) ) {
            std::string msg = "Inconsistent batch dimensions";
            throw std::invalid_argument(msg);
        }
    }
    // std::cout << "batch size " << batch_size << " num_samples " << batchsize_ << std::endl;
    if ( batch_size == 1 && batchsize_ != -1) {
        batch_size = batchsize_;
    } else if (batch_size == batchsize_ || batchsize_ == -1) {
        ;
    } else {
        std::string msg = "Num samples inconsistent with batch dimensions";
        throw std::invalid_argument(msg);
    }
    return(batch_size);
}

std::vector<Eigen::Map<vec>> ut::misc::map_array(np_array_c array, int batch_size, int base_dim) {
    // reshape if batch dim is missing
    if (array.ndim() == base_dim) {
        std::vector<int> shape = {1};
        for (int i = 0; i < base_dim; i++) {
            shape.push_back(array.shape(i));
        }
        array.resize(shape);
    } else if (array.ndim() == base_dim + 1) {
        ;
    } else {
        std::string msg = "Dimension mismatch during array map";
        throw std::invalid_argument(msg);
    }
    // check batch size
    if ((array.shape(0) != batch_size) && (array.shape(0) != 1) ) {
        std::string msg = "Batch shape mismatch during array map";
        throw std::invalid_argument(msg);
    }
    // map arrays
    std::vector<Eigen::Map<vec>> vectors;
    for (int i = 0; i < batch_size; i++) {
        int index = (array.shape(0) == batch_size) ? i : 0;
        np_array_c array_i = np_array_c(array[pybind11::slice(index, index+1, 1)]); 
        vectors.push_back(Eigen::Map<vec>((double*) array_i.data(), array_i.size()));
    }
    return(vectors);
}

std::vector<Eigen::Map<vec>> ut::misc::map_array(np_array_c array, np_array_c map, int base_dim) {
    // reshape if batch dim is missing
    if (array.ndim() == base_dim) {
        std::vector<int> shape = {1};
        for (int i = 0; i < base_dim; i++) {
            shape.push_back(array.shape(i));
        }
        array.resize(shape);
    } else if (array.ndim() == base_dim + 1) {
        ;
    } else {
        std::string msg = "Dimension mismatch during array map";
        throw std::invalid_argument(msg);
    }
    // map arrays
    int batch_size = map.size();
    std::vector<Eigen::Map<vec>> vectors;
    for (int i = 0; i < batch_size; i++) {
        int index = int(map.at(i));
        np_array_c array_i = np_array_c(array[pybind11::slice(index, index+1, 1)]); 
        vectors.push_back(Eigen::Map<vec>((double*) array_i.data(), array_i.size()));
    }
    return(vectors);
}

std::vector<Eigen::Map<vec>> ut::misc::parse_array_list(pybind11::list arrays, int batch_size) {
    // perform batch checks
    if (batch_size == -1) {
        batch_size = arrays.size();
    }
    if ((arrays.size() != batch_size) && (arrays.size() != 1) ) {
        std::string msg = "Dimension mismatch during array list parsing";
        throw std::invalid_argument(msg);
    }
    std::vector<Eigen::Map<vec>> vectors;
    for (int i = 0; i < batch_size; i++) {
        np_array_c array_i;
        if (arrays.size() == batch_size) {
            array_i = arrays[i].cast<np_array_c>();
        } else {
            array_i = arrays[0].cast<np_array_c>();
        }
        vectors.push_back(Eigen::Map<vec>((double*) array_i.data(), array_i.size()));
    }
    return(vectors);
}