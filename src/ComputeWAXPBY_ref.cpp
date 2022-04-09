
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
 @file ComputeWAXPBY_ref.cpp

 HPCG routine
 */

#include "ComputeWAXPBY_ref.hpp"
#ifndef HPCG_NO_OPENMP
 #include <omp.h>
#endif
#ifdef HPCG_WITH_CUDA
 #include <cuda_runtime.h>
 #include "cublas_v2.h"

 #if defined(HPCG_DEBUG) & !defined(HPCG_NO_MPI)
 #include <mpi.h>
 #include "Hpgmp_Params.hpp"
 #include "Utils_MPI.hpp"
 #endif
#endif
#include <cassert>
/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This is the reference WAXPBY impmentation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY
*/
template<class VectorX_type, class VectorY_type, class VectorW_type>
int ComputeWAXPBY_ref(const local_int_t n,
                      const typename VectorX_type::scalar_type alpha,
                      const VectorX_type & x,
                      const typename VectorY_type::scalar_type beta,
                      const VectorY_type & y,
                            VectorW_type & w) {
  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  // quick return
  if (n <= 0) return 0;

  typedef typename VectorX_type::scalar_type scalarX_type;
  typedef typename VectorY_type::scalar_type scalarY_type;
  typedef typename VectorW_type::scalar_type scalarW_type;

  scalarX_type * const xv = x.values;
  scalarY_type * const yv = y.values;
  scalarW_type * const wv = w.values;


#if !defined(HPCG_WITH_CUDA) | defined(HPCG_DEBUG)
  if (alpha==1.0) {
    #ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
    #endif
    for (local_int_t i=0; i<n; i++) wv[i] = xv[i] + beta * yv[i];
  } else if (beta==1.0) {
    #ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
    #endif
    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + yv[i];
  } else  {
    #ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
    #endif
    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + beta * yv[i];
  }
#endif

#ifdef HPCG_WITH_CUDA
  scalarX_type * const d_xv = x.d_values;
  scalarY_type * const d_yv = y.d_values;
  scalarW_type * const d_wv = w.d_values;

  // Only uniform-precision supported
  if ((std::is_same<scalarX_type, double>::value && std::is_same<scalarY_type, double>::value && std::is_same<scalarW_type, double>::value) ||
      (std::is_same<scalarX_type, float >::value && std::is_same<scalarY_type, float >::value && std::is_same<scalarW_type, float >::value)) {

    // Compute axpy on Nvidia GPU
    // w = x (assuming y is not w)
    if (cudaSuccess != cudaMemcpy(d_wv, d_xv, n*sizeof(scalarW_type), cudaMemcpyDeviceToDevice)) {
      printf( " Failed to memcpy d_w\n" );
    }
    if (std::is_same<scalarX_type, double>::value) {
      // w = alpha*w
      if (CUBLAS_STATUS_SUCCESS != cublasDscal (w.handle, n, (const double*)&alpha, (double*)d_wv, 1)) {
        printf( " Failed cublasDscal\n" );
      }
      // w += alpha*x
      if (CUBLAS_STATUS_SUCCESS != cublasDaxpy (w.handle, n, (const double*)&beta, (double*)d_yv, 1, (double*)d_wv, 1)) {
        printf( " Failed cublasDdot\n" );
      }
    } else if (std::is_same<scalarX_type, float>::value) {
      // w = beta*y
      if (CUBLAS_STATUS_SUCCESS != cublasSscal (w.handle, n, (const float*)&alpha, (float*)d_wv, 1)) {
        printf( " Failed cublasSscal\n" );
      }
      // w += alpha*x
      if (CUBLAS_STATUS_SUCCESS != cublasSaxpy (w.handle, n, (const float*)&beta, (float*) d_yv, 1, (float*) d_wv, 1)) {
        printf( " Failed cublasDdot\n" );
      }
    }

    #ifdef HPCG_DEBUG
    scalarW_type * tv = (scalarW_type *)malloc(n * sizeof(scalarW_type));
    if (cudaSuccess != cudaMemcpy(tv, d_wv, n*sizeof(scalarW_type), cudaMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_w\n" );
    }
    scalarW_type l_enorm = 0.0;
    scalarW_type l_wnorm = 0.0;
    scalarW_type l_xnorm = 0.0;
    scalarW_type l_ynorm = 0.0;
    for (int j=0; j<n; j++) {
      l_enorm += (tv[j]-wv[j])*(tv[j]-wv[j]);
      l_wnorm += wv[j]*wv[j];
      l_xnorm += xv[j]*xv[j];
      l_ynorm += yv[j]*yv[j];
    }
    scalarW_type enorm = 0.0;
    scalarW_type wnorm = 0.0;
    scalarX_type xnorm = 0.0;
    scalarY_type ynorm = 0.0;
    #ifndef HPCG_NO_MPI
    MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalarW_type>::getType ();
    MPI_Allreduce(&l_enorm, &enorm, 1, MPI_SCALAR_TYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&l_wnorm, &wnorm, 1, MPI_SCALAR_TYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&l_xnorm, &xnorm, 1, MPI_SCALAR_TYPE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&l_ynorm, &ynorm, 1, MPI_SCALAR_TYPE, MPI_SUM, MPI_COMM_WORLD);
    #else
    enorm = l_enorm;
    wnorm = l_wnorm;
    xnorm = l_xnorm;
    ynorm = l_ynorm;
    #endif
    enorm = sqrt(enorm);
    wnorm = sqrt(wnorm);
    xnorm = sqrt(xnorm);
    ynorm = sqrt(ynorm);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      HPCG_fout << rank << " : WAXPBY(" << n << "): error = " << enorm << " (alpha=" << alpha << ", beta=" << beta
	        << ", x=" << xnorm << ", y=" << ynorm << ", w=" << wnorm << ")" << std::endl;
    }
    free(tv);
    #endif
  } else {
    HPCG_fout << " Mixed-precision WAXPBY not supported" << std::endl;

    // copy Input vectors to Host
    if (cudaSuccess != cudaMemcpy(xv, d_xv, n*sizeof(scalarX_type), cudaMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_x\n" );
    }
    if (cudaSuccess != cudaMemcpy(yv, d_yv, n*sizeof(scalarY_type), cudaMemcpyDeviceToHost)) {
      printf( " Failed to memcpy d_w\n" );
    }

    // WAXPBY on Host
    for (local_int_t i=0; i<n; i++) wv[i] = alpha * xv[i] + beta * yv[i];

    // Copy output vector to Device
    if (cudaSuccess != cudaMemcpy(d_wv, wv, n*sizeof(scalarW_type), cudaMemcpyHostToDevice)) {
      printf( " Failed to memcpy d_w\n" );
    }
  }
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeWAXPBY_ref< Vector<double>, Vector<double>, Vector<double> >(int, double, Vector<double> const&, double, Vector<double> const&, Vector<double>&);

template
int ComputeWAXPBY_ref< Vector<float>, Vector<float>, Vector<float> >(int, float, Vector<float> const&, float, Vector<float> const&, Vector<float>&);


// mixed
template
int ComputeWAXPBY_ref< Vector<double>, Vector<float>, Vector<double> >(int, double, Vector<double> const&, float, Vector<float> const&, Vector<double>&);

