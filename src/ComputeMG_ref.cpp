
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
 @file ComputeSYMGS_ref.cpp

 HPCG routine
 */

#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeGS_Forward_ref.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#ifdef HPCG_DEBUG
#include "Hpgmp_Params.hpp"
#endif
#include <cassert>
#include <iostream>

/*!

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG
*/
template<class SparseMatrix_type, class Vector_type>
int ComputeMG_ref(const SparseMatrix_type & A, const Vector_type & r, Vector_type & x, bool symmetric) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    #if defined(HPCG_WITH_CUDA) & defined(HPCG_DEBUG)
    if (A.geom->rank==0) HPCG_fout << std::endl << " > PreSmooth( " << numberOfPresmootherSteps << " ) " << std::endl;
    #endif
    if (symmetric) {
      for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    } else {
      for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeGS_Forward_ref(A, r, x);
    }

    if (ierr!=0) return ierr;
    ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;

    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);  if (ierr!=0) return ierr;
    ierr = ComputeMG_ref(*A.Ac,*A.mgData->rc, *A.mgData->xc, symmetric);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation_ref(A, x);  if (ierr!=0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    #if defined(HPCG_WITH_CUDA) & defined(HPCG_DEBUG)
    if (A.geom->rank==0) HPCG_fout << " > PostSmooth( " << numberOfPostsmootherSteps << " ) " << std::endl;
    #endif
    if (symmetric) {
      for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    } else {
      for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeGS_Forward_ref(A, r, x);
    }
    if (ierr!=0) return ierr;
  }
  else {
    #if defined(HPCG_WITH_CUDA) & defined(HPCG_DEBUG)
    if (A.geom->rank==0) HPCG_fout << std::endl << " > CoarseSolve( " << 1 << " ) " << std::endl;
    #endif
    if (symmetric) {
      ierr = ComputeSYMGS_ref(A, r, x);
    } else {
      ierr = ComputeGS_Forward_ref(A, r, x);
    }
    if (ierr!=0) return ierr;
  }
  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int ComputeMG_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double> const&, Vector<double> const&, Vector<double>&, bool);

template
int ComputeMG_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float> const&, Vector<float> const&, Vector<float>&, bool);
