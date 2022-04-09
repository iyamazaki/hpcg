
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
 @file Vector.hpp

 HPCG data structures for dense vectors
 */

#ifndef MULTIVECTOR_HPP
#define MULTIVECTOR_HPP

#if defined(HPCG_WITH_CUDA)
 #include <cuda_runtime.h>
 #include <cublas_v2.h>
#elif defined(HPCG_WITH_HIP)
 #include <hip/hip_runtime_api.h>
 #include <rocblas.h>
#endif
#include <cassert>
#include <cstdlib>
#include "Vector.hpp"

template<class SC>
class MultiVector {
public:
  typedef SC scalar_type;
  local_int_t n;            //!< number of vectors
  local_int_t localLength;  //!< length of local portion of the vector
  SC * values;              //!< array of values
  #if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  SC * d_values;   //!< array of values
  #if defined(HPCG_WITH_CUDA)
  cublasHandle_t handle;
  #elif defined(HPCG_WITH_HIP)
  rocblas_handle handle;
  #endif
  #endif
  /*!
   This is for storing optimized data structures created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;
};

/*!
  Initializes input vectors.

  @param[in] V
  @param[in] localLength Length of local portion of input vector
  @param[in] n           Number of columns
 */
template<class MultiVector_type>
inline void InitializeMultiVector(MultiVector_type & V, local_int_t localLength, local_int_t n) {

  typedef typename MultiVector_type::scalar_type scalar_type;

  V.localLength = localLength;
  V.n = n;
  V.values = new scalar_type[localLength * n];
  #if defined(HPCG_WITH_CUDA)
  if (CUBLAS_STATUS_SUCCESS != cublasCreate(&V.handle)) {
    printf( " InitializeVector :: Failed to create Handle\n" );
  }
  if (cudaSuccess != cudaMalloc ((void**)&V.d_values, (localLength*n)*sizeof(scalar_type))) {
    printf( " InitializeVector :: Failed to allocate d_values\n" );
  }
  #elif defined(HPCG_WITH_HIP)
  if (rocblas_status_success != rocblas_create_handle(&V.handle)) {
    printf( " InitializeMultiVector :: Failed to create Handle\n" );
  }
  if (hipSuccess != hipMalloc ((void**)&V.d_values, (localLength*n)*sizeof(scalar_type))) {
    printf( " InitializeMultiVector :: Failed to allocate d_values\n" );
  }
  #endif
  V.optimizationData = 0;
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
template<class MultiVector_type>
inline void ZeroMultiVector(MultiVector_type & V) {

  typedef typename MultiVector_type::scalar_type scalar_type;

  local_int_t n = V.n;
  local_int_t m = V.localLength;
  scalar_type * vv = V.values;
  for (int i=0; i<m*n; ++i)
    vv[i] = 0.0;
  return;
}

/*!
  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
template<class MultiVector_type>
inline void GetMultiVector(MultiVector_type & V, local_int_t j1, local_int_t j2, MultiVector_type & Vj) {
  Vj.n = j2-j1+1;
  Vj.localLength = V.localLength;
  Vj.values = &V.values[V.localLength*j1];
  #if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  Vj.d_values = &V.d_values[V.localLength*j1];
  Vj.handle = V.handle;
  #endif
  return;
}

/*!
  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
template<class MultiVector_type, class Vector_type>
inline void GetVector(MultiVector_type & V, local_int_t j, Vector_type & vj) {
  vj.localLength = V.localLength;
  vj.values = &V.values[V.localLength*j];
  #if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  vj.d_values = &V.d_values[V.localLength*j];
  vj.handle = V.handle;
  #endif
  return;
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
template<class MultiVector_type>
inline void DeleteMultiVector(MultiVector_type & V) {

  delete [] V.values;
  V.localLength = 0;
  #if defined(HPCG_WITH_CUDA)
  cudaFree (V.d_values);
  cublasDestroy(V.handle);
  #elif defined(HPCG_WITH_HIP)
  hipFree(V.d_values);
  rocblas_destroy_handle(V.handle);
  #endif
  V.n = 0;
  return;
}

#endif // MULTIVECTOR_HPP

