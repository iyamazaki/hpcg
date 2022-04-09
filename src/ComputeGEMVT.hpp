
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeGEMVT.hpp

 HPCG data structures for dense vectors
 */

#ifndef COMPUTE_GEMVT_HPP
#define COMPUTE_GEMVT_HPP

#include "Geometry.hpp"
#include "MultiVector.hpp"
#include "Vector.hpp"
#include "SerialDenseMatrix.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMVT(const local_int_t m, const local_int_t n,
                const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const Vector_type & x,
                const typename      Vector_type::scalar_type beta,  const SerialDenseMatrix_type & y,
                bool & isOptimized);

#endif // COMPUTE_GEMVT
