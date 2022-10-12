#pragma once

// standard library
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
// pybind
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>


// function types
typedef double (*ArrayFun)(double*);

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

