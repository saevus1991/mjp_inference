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