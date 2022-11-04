#pragma once

#include <algorithm>

#include "../../types.h"


namespace ut::misc {

template <class T>
void print_vector(const std::vector<T>& input, std::string msg) {
    std::cout << msg << " ";
    for (int i = 0; i < input.size(); i ++) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

template <class T>
std::vector<T> set_combine(const std::vector<T>& first, const std::vector<T>& second) {
    std::vector<T> new_vec(first);
    for (int i = 0; i < second.size(); i++) {
        if ( std::find(first.begin(), first.end(), second[i]) == first.end() ) {
            new_vec.push_back(second[i]);
        }
    }
    std::sort(new_vec.begin(), new_vec.end());
    return(new_vec);
}

template <class T>
std::vector<T> set_combine(const std::vector<std::vector<T>>& set_list, const std::vector<unsigned>& index_list) {
    std::vector<T> new_vec;
    for (unsigned i = 0; i < index_list.size(); i++) {
        unsigned index = index_list[i];
        for (unsigned j = 0; j < set_list[index].size(); j++) {
            if ( std::find(new_vec.begin(), new_vec.end(), set_list[index][j]) == new_vec.end() ) {
                new_vec.push_back(set_list[index][j]);
            }
        }
    }
    std::sort(new_vec.begin(), new_vec.end());
    return(new_vec);
}

template <class T>
std::vector<T> set_combine(const std::vector<std::vector<T>>& set_list) {
    std::vector<unsigned> index_list(set_list.size());
    for (int i = 0; i < index_list.size(); i++) {
        index_list[i] = i;
    }
    return(set_combine(set_list, index_list));
} 

template <class T>
std::vector<T> set_intersect(const std::vector<T>& first, const std::vector<T>& second) {
    std::vector<T> intersect;
    for (int i = 0; i < first.size(); i++) {
        if ( std::find(second.begin(), second.end(), first[i]) != second.end() ) {
            intersect.push_back(first[i]);
        }
    }
    std::sort(intersect.begin(), intersect.end());
    return(intersect);
}

template <class T>
std::vector<T> set_minus(const std::vector<T>& first, const std::vector<T>& second) {
    std::vector<T> reduced;
    for (int i = 0; i < first.size(); i++) {
        if ( std::find(second.begin(), second.end(), first[i]) == second.end() ) {
            reduced.push_back(first[i]);
        }
    }
    std::sort(reduced.begin(), reduced.end());
    return(reduced);
}

template <class T> 
bool have_common_element(const std::vector<T>& first, const std::vector<T>& second) {
    return (std::find_first_of(first.begin(), first.end(), second.begin(), second.end()) != first.end());
}

template <class T>
bool is_subset(const std::vector<T>& first, const std::vector<T>& second) {
    bool is_subs = true;
    for (int i = 0; i < first.size(); i++) {
        is_subs = is_subs & (std::find(second.begin(), second.end(), first[i]) != second.end());
        if ( !is_subs ) {
            break;
        }
    }
    return(is_subs);
}

template <class T>
std::vector<std::vector<T>> prune_collection(const std::vector<std::vector<T>>& collection) {
    std::vector<std::vector<T>> pruned;
    for (int i = 0; i < collection.size(); i++) {
        bool is_maximal = true;
        for (int j = 0; j < pruned.size(); j++) {
            is_maximal = is_maximal & !is_subset(collection[i], pruned[j]);
        }
        if (is_maximal) {
            pruned.push_back(collection[i]);
        }
    }
    return(pruned);
}

template <class S, class T>
std::vector<S> sort_by(const std::vector<S>& first, const std::vector<T>& second) {
    // make range
    std::vector<unsigned> indices(first.size());
    std::iota(indices.begin(), indices.end(), 0);
    // sort indices by second
    std::sort(indices.begin(), indices.end(), [&] (unsigned i, unsigned j) -> bool { return(second[i] < second[j]);});
    // build output
    std::vector<S> sorted(first.size());
    for (unsigned i = 0; i < first.size(); i++) {
        sorted[i] = first[indices[i]];
    }
    return(sorted);
}

int infer_batchsize(std::vector<pybind11::buffer_info>& buffers, const std::vector<int>& base_dims, int batchsize_ = -1);

std::vector<Eigen::Map<vec>> map_array(np_array_c array, int batch_size, int base_dim);
std::vector<Eigen::Map<vec>> map_array(np_array_c array, np_array_c map, int base_dim);
std::vector<Eigen::Map<vec>> parse_array_list(pybind11::list arrays, int batch_size);


} // end namespace