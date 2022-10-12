#pragma once

#include "../../types.h"

namespace ut::math{

template <class T>
vec cumsum(T& x) {
    double* x_ptr = x.data();
    int x_size = x.rows();
    vec y(x_size);
    double* y_ptr = y.data();
    double tmp = 0.0;
    for (int i = 0; i < x_size; i++) {
        tmp += x_ptr[i];
        y_ptr[i] = tmp;
    }
    return(y);
}

} // end ut::math namespace