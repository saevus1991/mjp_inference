#pragma once

#include "../../types.h"


namespace ut {

inline void copy_buffer(const double* in, double* out, int numel) {
    for (int i = 0; i < numel; i++) {
        out[i] = in[i];
    }
    return;
}

inline np_array vec2array(vec& x) {
    int x_size = x.rows();
    double* x_ptr = x.data();
    np_array y(x_size);
    double *y_ptr = (double*) y.data();
    for (int i = 0; i < x_size; i++) {
        y_ptr[i] = x_ptr[i];
    }
    return(y);
}



// inline np_array vec2array(vec&& x) {
//     int x_size = x.rows();
//     double* x_ptr = x.data();
//     np_array y(x_size);
//     double *y_ptr = (double*) y.data();
//     for (int i = 0; i < x_size; i++) {
//         y_ptr[i] = x_ptr[i];
//     }
//     return(y);
// }

template <class T>
inline np_array vec2array(T&& x) {
    int x_size = x.rows();
    double* x_ptr = x.data();
    np_array y(x_size);
    double *y_ptr = (double*) y.data();
    for (int i = 0; i < x_size; i++) {
        y_ptr[i] = x_ptr[i];
    }
    return(y);
}

template <class T>
inline np_array mat2array(T&& x) {
    double* x_ptr = x.data();
    np_array y({x.rows(), x.cols()});
    double *y_ptr = (double*) y.data();
    for (int i = 0; i < x.size(); i++) {
        y_ptr[i] = x_ptr[i];
    }
    return(y);
}

inline vec vec2vec(const std::vector<double>& x) {
    vec y(x.size());
    copy_buffer(x.data(), y.data(), x.size());
    return(y);
}

inline vec vec2vec(std::vector<double>&& x) {
    vec y(x.size());
    copy_buffer(x.data(), y.data(), x.size());
    return(y);
}

inline np_array vec2array(std::vector<double>& x) {
    int x_size = x.size();
    np_array y(x_size);
    double *y_ptr = (double*) y.data();
    for (int i = 0; i < x_size; i++) {
        y_ptr[i] = x[i];
    }
    return(y);
}

inline np_array vec2array(std::vector<int>& x) {
    int x_size = x.size();
    np_array y(x_size);
    double *y_ptr = (double*) y.data();
    for (int i = 0; i < x_size; i++) {
        y_ptr[i] = double(x[i]);
    }
    return(y);
}

inline vec double2vec(double x) {
    vec y(1);
    y[0] = x;
    return(y);
}


template <class Key, class Val>
std::map<Key, Val> dict2map(pybind11::dict dict) {
    std::map<Key, Val> map;
    for (auto item : dict) {
        Key key = item.first.cast<Key>();
        map[key] = item.second.cast<Val>();
    }
    return(map);
}

template <class Key, class Val>
pybind11::dict map2dict(std::map<Key, Val>& map) {
    pybind11::dict dict;
    for (auto item : map) {
        std::string key(item.first);
        const char* key_ch = key.c_str();
        dict[key_ch] = item.second;
    }
    return(dict);
}

template <class Key, class Val>
pybind11::dict vec2dict(std::vector<Key>& key_vec, std::vector<Val>& val_vec) {
    pybind11::dict dict;
    unsigned size = key_vec.size();
    for (unsigned i = 0; i < size; i++) {
        std::string key(key_vec[i]);
        const char* key_ch = key.c_str();
        dict[key_ch] = static_cast<Val>(val_vec[i]);
    }
    return(dict);
}

template <class Key, class Val>
pybind11::dict vec2pointerdict(std::vector<Key>& key_vec, std::vector<Val>& val_vec) {
    pybind11::dict dict;
    unsigned size = key_vec.size();
    for (unsigned i = 0; i < size; i++) {
        std::string key(key_vec[i]);
        const char* key_ch = key.c_str();
        dict[key_ch] = &val_vec[i];
    }
    return(dict);
}

template <class Key>
std::vector<Key> extract_dict_keys(pybind11::dict dict) {
    std::vector<Key> key_list;
    for (auto item : dict) {
        Key key = item.first.cast<Key>();
        key_list.push_back(key);
    }
    return(key_list);
}

template <class Val>
std::vector<Val> extract_dict_values(pybind11::dict dict) {
    std::vector<Val> val_list;
    for (auto item : dict) {
        Val val = item.second.cast<Val>();
        val_list.push_back(val);
    }
    return(val_list);
}

template <class S, class T>
inline std::vector<S> lin2state(int ind, std::vector<T>& dim) {
    std::vector<S> res(dim.size());
    int rem = ind;
    for (int i = dim.size()-1; i >= 0 ; i--) {
        int div = dim[i];
        res[i] = rem % div;
        rem = rem / div;
        if (rem == 0)
            break;
    }
    return(res);
}

template <class S, class T>
inline int state2lin(std::vector<S>& ind, std::vector<T>& dim) {
    int base = 1;
    int res = 0;
    for (int i = ind.size()-1; i >= 0 ; i--) {
        res += int(ind[i])*base;
        base *= dim[i];
    }
    return(res);
}

template <class FUNC>
FUNC get_pyfunction(pybind11::tuple callable) {
    if (callable.size() > 0 ) {
        pybind11::capsule capsule = callable[0];
        return(reinterpret_cast<FUNC>(capsule.get_pointer()));
    } else {
        return(nullptr);
    }
}

} // end ut namespace
