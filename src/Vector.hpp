
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

#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <fstream>
#include <cassert>
#include <cstdlib>

#ifdef HPCG_WITH_BLAS
 #include "cblas.h"
#endif
#ifndef HPCG_NO_MPI
 #include <mpi.h>
#endif
#ifdef HPCG_WITH_CUDA
 #include <cuda_runtime.h>
 #include <cublas_v2.h>
#elif defined(HPCG_WITH_HIP)
 #include <hip/hip_runtime_api.h>
 #include <rocblas.h>
#endif

#include "DataTypes.hpp"
#include "hpgmp.hpp"
#include "Geometry.hpp"

template<class SC = double>
class Vector {
public:
  typedef SC scalar_type;
  local_int_t localLength;  //!< length of local portion of the vector
  SC * values;     //!< array of values

  // communicator
  comm_type comm;
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
  double time1, time2;
};

/*!
  Initializes input vector.

  @param[in] v
  @param[in] localLength Length of local portion of input vector
 */
template<class Vector_type>
inline void InitializeVector(Vector_type & v, local_int_t localLength, comm_type comm) {
  typedef typename Vector_type::scalar_type scalar_type;
  v.localLength = localLength;
  v.values = new scalar_type[localLength];
  v.comm = comm;
  #if defined(HPCG_WITH_CUDA)
  if (CUBLAS_STATUS_SUCCESS != cublasCreate(&v.handle)) {
    printf( " InitializeVector :: Failed to create Handle\n" );
  }
  if (cudaSuccess != cudaMalloc ((void**)&v.d_values, localLength*sizeof(scalar_type))) {
    printf( " InitializeVector :: Failed to allocate d_values\n" );
  }
  #elif defined(HPCG_WITH_HIP)
  if (rocblas_status_success != rocblas_create_handle(&v.handle)) {
    printf( " InitializeVector :: Failed to create Handle\n" );
  }
  if (hipSuccess != hipMalloc ((void**)&v.d_values, localLength*sizeof(scalar_type))) {
    printf( " InitializeVector :: Failed to allocate d_values\n" );
  }
  #endif
  v.optimizationData = 0;
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
template<class Vector_type>
inline void ZeroVector(Vector_type & v) {
  typedef typename Vector_type::scalar_type scalar_type;
  const int zero (0);

  local_int_t localLength = v.localLength;
  scalar_type * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = zero;
  #ifdef HPCG_WITH_CUDA
  if (cudaSuccess != cudaMemset(v.d_values, zero, localLength*sizeof(scalar_type))) {
    printf( " CopyVector :: Failed to memcpy d_v\n" );
  }
  #elif defined(HPCG_WITH_HIP)
  if (hipSuccess != hipMemset(v.d_values, zero, localLength*sizeof(scalar_type))) {
    printf( " CopyVector :: Failed to memcpy d_v\n" );
  }
  #endif
  return;
}
/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
template<class Vector_type>
inline void ScaleVectorValue(Vector_type & v, local_int_t index, typename Vector_type::scalar_type value) {
  typedef typename Vector_type::scalar_type scalar_type;
  assert(index>=0 && index < v.localLength);
  scalar_type * vv = v.values;
  vv[index] *= value;
  return;
}
/*!
  Multiply (scale) the vector by a given value.

  @param[inout] v Vector to be modified
  @param[in] value Value to scale by
 */
template<class Vector_type, class scalar_type = typename Vector_type::scalar_type>
inline void ScaleVectorValue(Vector_type & v, scalar_type value) {
  typedef typename Vector_type::scalar_type vector_scalar_type;
  local_int_t localLength = v.localLength;

#if defined(HPCG_WITH_KOKKOSKERNELS)
    using execution_space = Kokkos::DefaultExecutionSpace;
    #if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
    Kokkos::View<vector_scalar_type *,  Kokkos::LayoutLeft, execution_space> v_view(v.d_values, localLength);
    #else
    Kokkos::View<vector_scalar_type *,  Kokkos::LayoutLeft, execution_space> v_view(v.values, localLength);
    #endif
    //KokkosBlas::scal<Vector_type, scalar_type, Vector_type>(v_view, value, v_view);
    KokkosBlas::scal(v_view, value, v_view);
#elif defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  scalar_type * d_vv = v.d_values;
  if (std::is_same<scalar_type, double>::value) {
    #ifdef HPCG_WITH_CUDA
    if (CUBLAS_STATUS_SUCCESS != cublasDscal (v.handle, localLength, (const double*)&value, (double*)d_vv, 1)) {
      printf( " Failed cublasDscal\n" );
    }
    #elif defined(HPCG_WITH_HIP)
    if (rocblas_status_success != rocblas_dscal (v.handle, localLength, (const double*)&value, (double*)d_vv, 1)) {
      printf( " Failed rocblas_dscal\n" );
    }
    #endif
  } else if (std::is_same<scalar_type, float>::value) {
    #ifdef HPCG_WITH_CUDA
    if (CUBLAS_STATUS_SUCCESS != cublasSscal (v.handle, localLength, (const float*)&value, (float*)d_vv, 1)) {
      printf( " Failed cublasSscal\n" );
    }
    #elif defined(HPCG_WITH_HIP)
    if (rocblas_status_success != rocblas_sscal (v.handle, localLength, (const float*)&value, (float*)d_vv, 1)) {
      printf( " Failed rocblas_sscal\n" );
    }
    #endif
  }
#else
  // host CPU
  vector_scalar_type * vv = v.values;
  #if defined(HPCG_WITH_BLAS)
  if (std::is_same<vector_scalar_type, double>::value) {
    cblas_dscal(localLength, value, (double *)vv, 1);
  } else if (std::is_same<vector_scalar_type, float>::value) {
    cblas_sscal(localLength, value, (float *)vv, 1);
  }
  #else
  const scalar_type zero (0.0);
  if (value == zero) {
    for (int i=0; i<localLength; ++i) vv[i] = zero;
  } else {
    for (int i=0; i<localLength; ++i) vv[i] = value * scalar_type(vv[i]);
  }
  #endif
#endif
  return;
}
/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
template<class Vector_type>
inline void FillRandomVector(Vector_type & v) {
  typedef typename Vector_type::scalar_type scalar_type;
  local_int_t localLength = v.localLength;
  scalar_type * vv = v.values;
  for (int i=0; i<localLength; ++i) vv[i] = rand() / (scalar_type)(RAND_MAX) + 1.0;
  return;
}
/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
template<class Vector_src, class Vector_dst>
inline void CopyVector(const Vector_src & v, Vector_dst & w) {
  typedef typename Vector_src::scalar_type scalar_src;
  typedef typename Vector_dst::scalar_type scalar_dst;
  local_int_t localLength = v.localLength;
  assert(w.localLength >= localLength);
  scalar_src * vv = v.values;
  scalar_dst * wv = w.values;
#if (!defined(HPCG_WITH_CUDA) & !defined(HPCG_WITH_HIP)) | defined(HPCG_DEBUG)
  #if defined(HPCG_WITH_BLAS)
  if (std::is_same<scalar_src, scalar_dst>::value) {
    if (std::is_same<scalar_src, double>::value) {
      cblas_dcopy(localLength, (const double *)vv, 1, (double *)wv, 1);
    } else if (std::is_same<scalar_src, float>::value) {
      cblas_scopy(localLength, (const float *)vv, 1, (float *)wv, 1);
    }
  } else
  #endif
  {
    for (int i=0; i<localLength; ++i) wv[i] = vv[i];
  }
#endif

#if defined(HPCG_WITH_KOKKOSKERNELS)
    using execution_space = Kokkos::DefaultExecutionSpace;
    #if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
    Kokkos::View<scalar_src *,  Kokkos::LayoutLeft, execution_space> v_view(v.d_values, localLength);
    Kokkos::View<scalar_dst *,  Kokkos::LayoutLeft, execution_space> w_view(w.d_values, localLength);
    #else
    Kokkos::View<scalar_src *,  Kokkos::LayoutLeft, execution_space> v_view(v.values, localLength);
    Kokkos::View<scalar_dst *,  Kokkos::LayoutLeft, execution_space> w_view(w.values, localLength);
    #endif
    const scalar_src one (1.0);
    KokkosBlas::scal(w_view, one, v_view);
#elif defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  if (std::is_same<scalar_src, scalar_dst>::value) {
    #ifdef HPCG_DEBUG
    HPCG_fout << " CopyVector ( Unit-precision )" << std::endl;
    #endif
    #if defined(HPCG_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(w.d_values, v.d_values, localLength*sizeof(scalar_src), cudaMemcpyDeviceToDevice)) {
      printf( " CopyVector :: Failed to memcpy d_x\n" );
    }
    #elif defined(HPCG_WITH_HIP)
    if (hipSuccess != hipMemcpy(w.d_values, v.d_values, localLength*sizeof(scalar_src), hipMemcpyDeviceToDevice)) {
      printf( " CopyVector :: Failed to memcpy d_x\n" );
    }
    #endif
  } else {
    #ifdef HPCG_DEBUG
    HPCG_vout << " CopyVector :: Mixed-precision not supported" << std::endl;
    #endif
    // Copy input vector to Host
    #if defined(HPCG_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(vv, v.d_values, localLength*sizeof(scalar_src), cudaMemcpyDeviceToHost)) {
      printf( " CopyVector :: Failed to memcpy d_v\n" );
    }
    #elif defined(HPCG_WITH_HIP)
    if (hipSuccess != hipMemcpy(vv, v.d_values, localLength*sizeof(scalar_src), hipMemcpyDeviceToHost)) {
      printf( " CopyVector :: Failed to memcpy d_v\n" );
    }
    #endif

    // Copy on Host
    for (int i=0; i<localLength; ++i) wv[i] = vv[i];

    // Copy output vector to Device
    #if defined(HPCG_WITH_CUDA)
    if (cudaSuccess != cudaMemcpy(w.d_values, wv, localLength*sizeof(scalar_dst), cudaMemcpyHostToDevice)) {
      printf( " CopyVector :: Failed to memcpy d_w\n" );
    }
    #elif defined(HPCG_WITH_HIP)
    if (hipSuccess != hipMemcpy(w.d_values, wv, localLength*sizeof(scalar_dst), hipMemcpyHostToDevice)) {
      printf( " CopyVector :: Failed to memcpy d_w\n" );
    }
    #endif
  }
#endif
  return;
}


/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
template<class Vector_type>
inline void DeleteVector(Vector_type & v) {

  delete [] v.values;
  #if defined(HPCG_WITH_CUDA)
  cudaFree(v.d_values);
  cublasDestroy(v.handle);
  #elif defined(HPCG_WITH_HIP)
  hipFree(v.d_values);
  rocblas_destroy_handle(v.handle);
  #endif
  v.localLength = 0;
  return;
}

#endif // VECTOR_HPP
