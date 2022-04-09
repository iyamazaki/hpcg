
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
 @file GenerateProblem.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "GenerateNonsymProblem.hpp"
#include "GenerateNonsymProblem_v1_ref.hpp"


/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

template<class SparseMatrix_type, class Vector_type>
void GenerateNonsymProblem(SparseMatrix_type & A, Vector_type * b, Vector_type * x, Vector_type * xexact, bool init_vect) {

  // The call to this reference version of GenerateProblem can be replaced with custom code.
  // However, the data structures must remain unchanged such that the CheckProblem function is satisfied.
  // Furthermore, any code must work for general unstructured sparse matrices.  Special knowledge about the
  // specific nature of the sparsity pattern may not be explicitly used.

  #if 1
  return GenerateNonsymProblem_v1_ref(A, b, x, xexact, init_vect);
  #else
  return GenerateNonsymProblem_ref(A, b, x, xexact, init_vect);
  #endif
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
void GenerateNonsymProblem< SparseMatrix<double>, Vector<double> >(SparseMatrix<double>&, Vector<double>*, Vector<double>*, Vector<double>*, bool);

template
void GenerateNonsymProblem< SparseMatrix<float>, Vector<float> >(SparseMatrix<float>&, Vector<float>*, Vector<float>*, Vector<float>*, bool);


// mixed
template
void GenerateNonsymProblem< SparseMatrix<float>, Vector<double> >(SparseMatrix<float>&, Vector<double>*, Vector<double>*, Vector<double>*, bool);

