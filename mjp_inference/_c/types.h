#pragma once

// standard library
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <random>
// pybind
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

// numpy types
typedef pybind11::array_t<double> np_array;
typedef pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> np_array_c;

// eigen types
typedef Eigen::VectorXd vec;
typedef Eigen::MatrixXd mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_rm;
typedef Eigen::SparseMatrix<double> csc_mat;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> csr_mat;
typedef Eigen::Triplet<double> triplet;

// function types
typedef double (*ArrayFun)(const double*); 
typedef void (*TransformFun)(double, const double*, const double*, double*);
typedef void (*TransformGrad)(double, const double*, const double*, const double*, double*);
typedef void (*Sampler)(double, const double*, const double*, const double*, double*); 
typedef double (*Llh)(double, const double*, const double*); 
typedef Eigen::Map<vec> (*ArrayGetter)(Eigen::Map<vec>&, int, int);
typedef std::function<vec (vec&)> Operator;
typedef double (*RVSampler)(std::mt19937* rng);

// numerical constants

#ifdef _WIN32
    #define M_PI 3.14159265358979323846
#endif

constexpr double min_double = std::numeric_limits<double>::min();
constexpr double max_double = std::numeric_limits<double>::max();