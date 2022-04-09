
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
 @file ComputeRestriction_ref.cpp

 HPCG routine
 */


#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeRestriction_ref.hpp"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeRestriction_ref(const SparseMatrix_type & A, const Vector_type & rf) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;

  scalar_type * Axfv = A.mgData->Axf->values;
  scalar_type * rfv = rf.values;
  scalar_type * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

  #ifdef HPCG_WITH_CUDA
   local_int_t n = rf.localLength;
   scalar_type * d_Axfv = A.mgData->Axf->d_values;
   scalar_type * d_rfv  = rf.d_values;
   scalar_type * d_rcv  = A.mgData->rc->d_values;
   #if 1
   const scalar_type zero ( 0.0);
   const scalar_type  one ( 1.0);
   const scalar_type mone (-1.0);
   if (std::is_same<scalar_type, double>::value) {
     if (CUSPARSE_STATUS_SUCCESS != cusparseDcsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   nc, n, nc,
                                                   (const double*)&one,  A.mgData->descrA,
                                                                         (double*)A.mgData->d_nzvals, A.mgData->d_row_ptr, A.mgData->d_col_idx,
                                                                         (double*)d_rfv,
                                                   (const double*)&zero, (double*)d_rcv)) {
       printf( " Failed cusparseDcsrmv\n" );
     }
     if (CUSPARSE_STATUS_SUCCESS != cusparseDcsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   nc, n, nc,
                                                   (const double*)&mone, A.mgData->descrA,
                                                                         (double*)A.mgData->d_nzvals, A.mgData->d_row_ptr, A.mgData->d_col_idx,
                                                                         (double*)d_Axfv,
                                                   (const double*)&one,  (double*)d_rcv)) {
       printf( " Failed cusparseDcsrmv\n" );
     }
   } else if (std::is_same<scalar_type, float>::value) {
     if (CUSPARSE_STATUS_SUCCESS != cusparseScsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   nc, n, nc,
                                                   (const float*)&one,  A.mgData->descrA,
                                                                        (float*)A.mgData->d_nzvals, A.mgData->d_row_ptr, A.mgData->d_col_idx,
                                                                        (float*)d_rfv,
                                                   (const float*)&zero, (float*)d_rcv)) {
       printf( " Failed cusparseScsrmv\n" );
     }
     if (CUSPARSE_STATUS_SUCCESS != cusparseScsrmv(A.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   nc, n, nc,
                                                   (const float*)&mone, A.mgData->descrA,
                                                                        (float*)A.mgData->d_nzvals, A.mgData->d_row_ptr, A.mgData->d_col_idx,
                                                                        (float*)d_Axfv,
                                                   (const float*)&one,  (float*)d_rcv)) {
       printf( " Failed cusparseScsrmv\n" );
     }
   }
   #else
   // Copy the whole prologated vector from Device to Host
   if (cudaSuccess != cudaMemcpy(rfv, d_rfv, n*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
     printf( " Failed to memcpy d_rfv\n" );
   }
   if (cudaSuccess != cudaMemcpy(Axfv, d_Axfv, n*sizeof(scalar_type), cudaMemcpyDeviceToHost)) {
     printf( " Failed to memcpy d_Axfv\n" );
   }

   // Restriction on CPU 
   if (A.geom->rank==0) printf( " Restriction on CPU\n" );
   for (local_int_t i=0; i<nc; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];

   // Copy the whole restricted vector from Host to Device
   if (cudaSuccess != cudaMemcpy(d_rcv, rcv, nc*sizeof(scalar_type), cudaMemcpyHostToDevice)) {
     printf( " Failed to memcpy d_x\n" );
   }
   #endif
  #else
   // host
   #ifndef HPCG_NO_OPENMP
   #pragma omp parallel for
   #endif
   for (local_int_t i=0; i<nc; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
  #endif

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeRestriction_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double> const&);

template
int ComputeRestriction_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float> const&);


