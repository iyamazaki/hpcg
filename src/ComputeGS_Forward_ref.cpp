
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
 @file ComputeGS_Forward_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
 #include "ExchangeHalo.hpp"
#endif
#ifdef HPCG_WITH_CUDA
 #include <cuda_runtime.h>
 #include <cublas_v2.h>
 #include "ComputeSPMV.hpp"
 #include "ComputeWAXPBY.hpp"
 #ifdef HPCG_DEBUG
 #include <mpi.h>
 #include "Utils_MPI.hpp"
 #include "Hpgmp_Params.hpp"
 #endif
#endif
#include "ComputeGS_Forward_ref.hpp"
#include <cassert>
#include <iostream>

/*!
  Computes one forward step of Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeGS_Forward
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeGS_Forward_ref(const SparseMatrix_type & A, const Vector_type & r, Vector_type & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  const local_int_t nrow = A.localNumberOfRows;
  const local_int_t ncol = A.localNumberOfColumns;

  const scalar_type * const rv = r.values;
  scalar_type * const xv = x.values;

#ifndef HPCG_NO_MPI
  #ifdef HPCG_WITH_CUDA
  // workspace
  Vector_type b = A.x; // nrow
  scalar_type * const d_bv = b.d_values;
  scalar_type * const d_xv = x.d_values;

  // Copy local part of X to HOST CPU
  if (A.geom->rank==0) printf( " HaloExchange on Host for GS_Forward\n" );
  if (cudaSuccess != cudaMemcpy(xv, d_xv, nrow*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_y\n" );
  }
  #endif

  // Exchange Halo on HOST CPU
  ExchangeHalo(A, x);

  #ifdef HPCG_WITH_CUDA
  // Copy X (after Halo Exchange on host) to device
  #define HPCG_COMPACT_GS
  #ifdef HPCG_COMPACT_GS
  // Copy non-local part of X (after Halo Exchange) into x0 on device
  //if (cudaSuccess != cudaMemcpy(&d_xv[nrow], &xv[nrow], (ncol-nrow)*sizeof(scalar_type), cudaMemcpyHostToDevice)) {
  //  printf( " Failed to memcpy d_y\n" );
  //}
  #else
  Vector_type x0 = A.y; // ncol
  scalar_type * const x0v = x0.values;
  CopyVector(x, x0); // this also copy on CPU, which is needed only for debug
  #endif

  #ifdef HPCG_DEBUG
  if (A.geom->rank==0) {
    HPCG_fout << A.geom->rank << " : ComputeGS(" << nrow << " x " << ncol << ") start" << std::endl;
  }
  #endif
  #endif
#endif

#if !defined(HPCG_WITH_CUDA) | defined(HPCG_DEBUG)
  scalar_type ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues

  for (local_int_t i=0; i < nrow; i++) {
    const scalar_type * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const scalar_type currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    scalar_type sum = rv[i]; // RHS value

    for (int j=0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
  }
#endif

#ifdef HPCG_WITH_CUDA
  const scalar_type  one ( 1.0);
  const scalar_type mone (-1.0);

  #ifdef HPCG_COMPACT_GS
  // b = r - Ux
  if (cudaSuccess != cudaMemcpy(d_bv, r.d_values, nrow*sizeof(scalar_type), cudaMemcpyDeviceToDevice)) {
    printf( " Failed to memcpy d_r\n" );
  }
  if (std::is_same<scalar_type, double>::value) {
     if (CUSPARSE_STATUS_SUCCESS != cusparseDcsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   nrow, ncol, A.nnzU,
                                                   (const double*)&mone,  A.descrU,
                                                                         (double*)A.d_Unzvals, A.d_Urow_ptr, A.d_Ucol_idx,
                                                                         (double*)d_xv,
                                                   (const double*)&one,  (double*)d_bv)) {
       printf( " Failed cusparseDcsrmv\n" );
     }
  } else if (std::is_same<scalar_type, float>::value) {
     if (CUSPARSE_STATUS_SUCCESS != cusparseScsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   nrow, ncol, A.nnzU,
                                                   (const float*)&mone, A.descrA,
                                                                        (float*)A.d_Unzvals, A.d_Urow_ptr, A.d_Ucol_idx,
                                                                        (float*)d_xv,
                                                   (const float*)&one,  (float*)d_bv)) {
       printf( " Failed cusparseScsrmv\n" );
     }
  }
  #else
  // b = r - Ax0
  ComputeSPMV(A, x0, b);
  ComputeWAXPBY(nrow, -one, b, one, r, b, A.isWaxpbyOptimized);
  #endif

  // x = L^{-1}b
  if (std::is_same<scalar_type, double>::value) {
     if (CUSPARSE_STATUS_SUCCESS != cusparseDcsrsv_solve(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nrow,
                                                         (const double*)&one, A.descrL,
                                                                              (double*)A.d_Lnzvals, A.d_Lrow_ptr, A.d_Lcol_idx,
                                                                              A.infoL,
                                                         (double*)d_bv, (double*)d_xv)) {
       printf( " Failed cusparseDcsrv_solve\n" );
     }
  } else if (std::is_same<scalar_type, float>::value) {
     if (CUSPARSE_STATUS_SUCCESS != cusparseScsrsv_solve(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nrow,
                                                         (const float*)&one, A.descrL,
                                                                             (float*)A.d_Lnzvals, A.d_Lrow_ptr, A.d_Lcol_idx,
                                                                             A.infoL,
                                                         (float*)d_bv, (float*)d_xv)) {
       printf( " Failed cusparseScsrv_solve\n" );
     }
  }

  #ifdef HPCG_DEBUG
  scalar_type * tv = (scalar_type *)malloc(nrow * sizeof(scalar_type));
  for (int i=0; i<nrow; i++) tv[i] = xv[i];
  // copy x to host for check inside WAXPBY (debug)
  if (cudaSuccess != cudaMemcpy(xv, d_xv, nrow*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
    printf( " Failed to memcpy d_b\n" );
  }
  #endif

  #ifndef HPCG_COMPACT_GS
  // x = x + x0
  ComputeWAXPBY(nrow, one, x, one, x0, x, A.isWaxpbyOptimized);
  #endif

  #ifdef HPCG_DEBUG
  scalar_type l_enorm = 0.0;
  scalar_type l_xnorm = 0.0;
  scalar_type l_rnorm = 0.0;
  for (int j=0; j<nrow; j++) {
    l_xnorm += tv[j]*tv[j];
  }
  for (int j=0; j<nrow; j++) {
    l_enorm += (xv[j]-tv[j])*(xv[j]-tv[j]);
    l_rnorm += rv[j]*rv[j];
  }
  scalar_type enorm = 0.0;
  scalar_type xnorm = 0.0;
  scalar_type rnorm = 0.0;
  #ifndef HPCG_NO_MPI
  MPI_Datatype MPI_SCALAR_TYPE = MpiTypeTraits<scalar_type>::getType ();
  MPI_Allreduce(&l_enorm, &enorm, 1, MPI_SCALAR_TYPE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&l_xnorm, &xnorm, 1, MPI_SCALAR_TYPE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&l_rnorm, &rnorm, 1, MPI_SCALAR_TYPE, MPI_SUM, MPI_COMM_WORLD);
  #else
  enorm = l_enorm;
  xnorm = l_xnorm;
  rnorm = l_rnorm;
  #endif
  enorm = sqrt(enorm);
  xnorm = sqrt(xnorm);
  rnorm = sqrt(rnorm);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    HPCG_fout << rank << " : GS_forward(" << nrow << " x " << ncol << "): error = " << enorm << " (x=" << xnorm << ", r=" << rnorm << ")" << std::endl;
  }
  free(tv);
  #endif
#endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeGS_Forward_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double> const&, Vector<double>&);

template
int ComputeGS_Forward_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float> const&, Vector<float>&);

