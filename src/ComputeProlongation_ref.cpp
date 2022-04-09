
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
 @file ComputeProlongation_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeProlongation_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeProlongation_ref(const SparseMatrix_type & Af, Vector_type & xf) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;

  scalar_type * xfv = xf.values;
  scalar_type * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

  #ifdef HPCG_WITH_CUDA
   local_int_t n = xf.localLength;
   scalar_type * d_xfv = xf.d_values;
   scalar_type * d_xcv = Af.mgData->xc->d_values;
   #if 1
   const scalar_type zero ( 0.0);
   const scalar_type  one ( 1.0);
   const scalar_type mone (-1.0);
   if (std::is_same<scalar_type, double>::value) {
     if (CUSPARSE_STATUS_SUCCESS != cusparseDcsrmv(Af.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                                   nc, n, nc,
                                                   (const double*)&one, Af.mgData->descrA,
                                                                        (double*)Af.mgData->d_nzvals, Af.mgData->d_row_ptr, Af.mgData->d_col_idx,
                                                                        (double*)d_xcv,
                                                   (const double*)&one, (double*)d_xfv)) {
       printf( " Failed cusparseDcsrmv\n" );
     }
   } else if (std::is_same<scalar_type, float>::value) {
     if (CUSPARSE_STATUS_SUCCESS != cusparseScsrmv(Af.cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                                   nc, n, nc,
                                                   (const float*)&one, Af.mgData->descrA,
                                                                       (float*)Af.mgData->d_nzvals, Af.mgData->d_row_ptr, Af.mgData->d_col_idx,
                                                                       (float*)d_xcv,
                                                   (const float*)&one, (float*)d_xfv)) {
       printf( " Failed cusparseScsrmv\n" );
     }
   }
   #else
   // Copy the whole compressed vector from Device to Host..
   if (Af.geom->rank==0) printf( " Prologation on CPU ..\n" );
   if (cudaSuccess != cudaMemcpy(xfv, d_xfv,  n*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
     printf( " Failed to memcpy d_x\n" );
   }
   if (cudaSuccess != cudaMemcpy(xcv, d_xcv, nc*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
     printf( " Failed to memcpy d_x\n" );
   }

   // Prologation on host
   for (local_int_t i=0; i<nc; ++i) xfv[f2c[i]] += xcv[i]; // This loop is safe to vectorize

   // Copy the whole expanded vector from Host to Device..
   if (cudaSuccess != cudaMemcpy(d_xfv, xfv, n*sizeof(scalar_type), cudaMemcpyHostToDevice)) {
     printf( " Failed to memcpy d_x\n" );
   }
   #endif
  #else
   #ifndef HPCG_NO_OPENMP
   #pragma omp parallel for
   #endif
   // TODO: Somehow note that this loop can be safely vectorized since f2c has no repeated indices
   for (local_int_t i=0; i<nc; ++i) xfv[f2c[i]] += xcv[i]; // This loop is safe to vectorize
  #endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeProlongation_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double>&);

template
int ComputeProlongation_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float>&);
