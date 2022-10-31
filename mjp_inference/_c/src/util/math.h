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

template <class T>
std::vector<unsigned> nn_project(T& state) {
  /*
  Set all elements below threshold to threshold and return corresponding indices
  */
  // preperations
  std::vector<unsigned> indices;
  indices.reserve(int(state.rows()));
  double* state_ptr = (double*) state.data();
  // iterate over components
  for (unsigned i = 0; i < state.rows(); i++) {
    if (state_ptr[i] < 1e-10) {
      state_ptr[i] = 1e-10;
      indices.push_back(i);
    }
  }
  return(indices);
}

template <class T>
void project_positive(T& array) {
  // get vars
  double* array_ptr = (double*) array.data();
  unsigned len = array.size();
  for (unsigned i = 0; i < len; i++) {
    if (array_ptr[i] < min_double) {
        array_ptr[i] = min_double;
    }
  }
}

inline double solve_quadratic(double a, double b, double c) {
    // Solve quadratix equation a*x^2+b*x+c == 0 and return the larger of the solutions (assuming one exists)
    double sol;
    if ( a == 0 )
        sol = -c/b;
    else {
        sol = -b+std::sqrt(b*b-4*a*c);
        sol /= 2*a;
    }
    return( sol );
}

} // end ut::math namespace