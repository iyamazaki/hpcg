
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
 @file ComputeDotProduct_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#include "Utils_MPI.hpp"
#endif
#ifndef HPCG_NO_OPENMP
 #include <omp.h>
#endif
#ifdef HPCG_WITH_CUDA
 #include <cuda_runtime.h>
 #include <cublas_v2.h>
#elif defined(HPCG_WITH_HIP)
 #include <hip/hip_runtime_api.h>
 #include <rocblas.h>
#endif

#include <cassert>
#include "ComputeDotProduct_ref.hpp"

#ifdef HPCG_DEBUG
#include "Hpgmp_Params.hpp"
#endif

/*!
  Routine to compute the dot product of two vectors where:

  This is the reference dot-product implementation.  It _CANNOT_ be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] x, y the input vectors
  @param[in] result a pointer to scalar value, on exit will contain result.
  @param[out] time_allreduce the time it took to perform the communication between processes

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
*/
template<class Vector_type>
int ComputeDotProduct_ref(const local_int_t n, const Vector_type & x, const Vector_type & y,
                          typename Vector_type::scalar_type & result, double & time_allreduce) {
  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  typedef typename Vector_type::scalar_type scalar_type;
  scalar_type local_result (0.0);

#if !defined(HPCG_WITH_CUDA) | defined(HPCG_DEBUG)
  scalar_type * xv = x.values;
  scalar_type * yv = y.values;
  if (yv==xv) {
    #ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result)
    #endif
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*xv[i];
  } else {
    #ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction (+:local_result)
    #endif
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*yv[i];
  }
#endif

#if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  scalar_type* d_x = x.d_values;
  scalar_type* d_y = y.d_values;

  #ifdef HPCG_DEBUG
  scalar_type local_tmp = local_result;
  #endif
  #if defined(HPCG_WITH_CUDA)
  // Compute dot on Nvidia GPU
  cublasHandle_t handle = x.handle;
  if (std::is_same<scalar_type, double>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasDdot (handle, n, (double*)d_x, 1, (double*)d_y, 1, (double*)&local_result)) {
      printf( " Failed cublasDdot\n" );
    }
  } else if (std::is_same<scalar_type, float>::value) {
    if (CUBLAS_STATUS_SUCCESS != cublasSdot (handle, n, (float*)d_x, 1,  (float*)d_y, 1,  (float*)&local_result)) {
      printf( " Failed cublasSdot\n" );
    }
  }
  #elif defined(HPCG_WITH_HIP)
  // Compute dot on AMD GPU
  rocblas_handle handle = x.handle;
  #if 1 // TODO remove this
  if (hipSuccess != hipMemcpy(d_x, xv, sizeof(scalar_type) * n, hipMemcpyHostToDevice)) {
    printf( " Failed hipMemcpy d_x\n" );
  }
  if (hipSuccess != hipMemcpy(d_y, yv, sizeof(scalar_type) * n, hipMemcpyHostToDevice)) {
    printf( " Failed hipMemcpy d_y\n" );
  }
  #endif
  if (std::is_same<scalar_type, double>::value) {
    if (rocblas_status_success != rocblas_ddot (handle, n, (double*)d_x, 1, (double*)d_y, 1, (double*)&local_result)) {
      printf( " Failed rocblas_ddot\n" );
    }
  } else if (std::is_same<scalar_type, float>::value) {
    if (rocblas_status_success != rocblas_sdot (handle, n, (float*)d_x, 1,  (float*)d_y, 1,  (float*)&local_result)) {
      printf( " Failed rocblas_sdot\n" );
    }
  }
  #endif
#endif

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType ();
  double t0 = mytimer();
  scalar_type global_result (0.0);
  MPI_Allreduce(&local_result, &global_result, 1, MPI_SCALAR_TYPE, MPI_SUM,
                MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;

  #if defined(HPCG_WITH_CUDA) & defined(HPCG_DEBUG)
  scalar_type global_tmp (0.0);
  MPI_Allreduce(&local_tmp, &global_tmp, 1, MPI_SCALAR_TYPE, MPI_SUM,
                MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    HPCG_fout << rank << " : DotProduct(" << n << "): error = " << global_tmp-global_result << " (dot=" << global_result << ")" << std::endl;
  }
  #endif
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeDotProduct_ref<Vector<double> >(int, Vector<double> const&, Vector<double> const&, double&, double&);

template
int ComputeDotProduct_ref<Vector<float> >(int, Vector<float> const&, Vector<float> const&, float&, double&);
