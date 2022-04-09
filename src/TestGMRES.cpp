
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
 @file TestGMRES.cpp

 HPCG routine
 */

// Changelog
//
// Version 0.4
// - Added timing of setup time for sparse MV
// - Corrected percentages reported for sparse MV with overhead
//
/////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
using std::endl;
#include <vector>
#include "Hpgmp_Params.hpp"

#include "TestGMRES.hpp"
#include "GMRES.hpp"
#include "GMRES_IR.hpp"
#include "mytimer.hpp"

/*!
  Test the correctness of the Preconditined CG implementation by using a system matrix with a dominant diagonal.

  @param[in]    geom The description of the problem's geometry.
  @param[in]    A    The known system matrix
  @param[in]    data the data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[out]   testcg_data the data structure with the results of the test including pass/fail information

  @return Returns zero on success and a non-zero value otherwise.

  @see CG()
 */


template<class SparseMatrix_type, class SparseMatrix_type2, class CGData_type, class CGData_type2, class Vector_type, class TestCGData_type>
int TestGMRES(SparseMatrix_type & A, SparseMatrix_type2 & A_lo, CGData_type & data, CGData_type2 & data_lo, Vector_type & b, Vector_type & x,
              TestCGData_type & testcg_data, bool test_diagonal_exaggeration, bool test_noprecond) {

  typedef typename SparseMatrix_type::scalar_type scalar_type;
  typedef typename SparseMatrix_type2::scalar_type scalar_type2;
  typedef Vector<scalar_type2> Vector_type2;

  // Use this array for collecting timing information
  double flops;
  std::vector< double > times(8,0.0);
  // Temporary storage for holding original diagonal and RHS
  Vector_type origDiagA, exaggeratedDiagA, origB;
  Vector_type2 origDiagA2, exagDiagA2;
  InitializeVector(origDiagA, A.localNumberOfRows);
  InitializeVector(origDiagA2, A_lo.localNumberOfRows);
  InitializeVector(exaggeratedDiagA, A.localNumberOfRows);
  InitializeVector(exagDiagA2, A_lo.localNumberOfRows);
  InitializeVector(origB, A.localNumberOfRows);
  CopyMatrixDiagonal(A, origDiagA);
  CopyMatrixDiagonal(A_lo, origDiagA2);
  CopyVector(origDiagA, exaggeratedDiagA);
  CopyVector(origDiagA2, exagDiagA2);
  CopyVector(b, origB);

  // TODO: This should be moved to somewhere-else, e.g., SetupProblem
  if (test_diagonal_exaggeration) {
    // Modify the matrix diagonal to greatly exaggerate diagonal values.
    // CG should converge in about 10 iterations for this problem, regardless of problem size
    if (A.geom->rank==0) HPCG_fout << std::endl << " ** applying diagonal exaggeration ** " << std::endl << std::endl;
    for (local_int_t i=0; i< A.localNumberOfRows; ++i) {
      global_int_t globalRowID = A.localToGlobalMap[i];
      if (globalRowID<9) {
        scalar_type scale = (globalRowID+2)*1.0e6;
        scalar_type2 scale2 = (globalRowID+2)*1.0e6;
        ScaleVectorValue(exaggeratedDiagA, i, scale);
        ScaleVectorValue(exagDiagA2, i, scale2);
        ScaleVectorValue(b, i, scale);
      } else {
        ScaleVectorValue(exaggeratedDiagA, i, 1.0e6);
        ScaleVectorValue(exagDiagA2, i, 1.0e6);
        ScaleVectorValue(b, i, 1.0e6);
      }
    }
    ReplaceMatrixDiagonal(A, exaggeratedDiagA);
    ReplaceMatrixDiagonal(A_lo, exagDiagA2);//TODO probably some funny casting here... need to do properly.
  } else {
    if (A.geom->rank==0) HPCG_fout << std::endl << " ** skippping diagonal exaggeration ** " << std::endl << std::endl;
  }

  int niters = 0;
  scalar_type normr (0.0);
  scalar_type normr0 (0.0);
  int restart_length = 30;
  int maxIters = 5000;
  int numberOfCgCalls = 2;
  scalar_type tolerance = 1.0e-12; // Set tolerance to reasonable value for grossly scaled diagonal terms
  testcg_data.expected_niters_no_prec = 12; // For the unpreconditioned CG call, we should take about 10 iterations, permit 12
  testcg_data.expected_niters_prec = 2;   // For the preconditioned case, we should take about 1 iteration, permit 2
  testcg_data.niters_max_no_prec = 0;
  testcg_data.niters_max_prec = 0;
  for (int k=(test_noprecond ? 0 : 1); k<2; ++k)
  { // This loop tests both unpreconditioned and preconditioned runs
    int expected_niters = testcg_data.expected_niters_no_prec;
    if (k==1) expected_niters = testcg_data.expected_niters_prec;
    for (int i=0; i< numberOfCgCalls; ++i) {
      ZeroVector(x); // Zero out x

      double time_tic = mytimer();
      int ierr = GMRES(A, data, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, &times[0], &flops, k==1);
      double time_solve = mytimer() - time_tic;
      if (ierr) HPCG_fout << "Error in call to GMRES: " << ierr << ".\n" << endl;
      if (niters <= expected_niters) {
        ++testcg_data.count_pass;
      } else {
        ++testcg_data.count_fail;
      }
      if (k==0 && niters > testcg_data.niters_max_no_prec) testcg_data.niters_max_no_prec = niters; // Keep track of largest iter count
      if (k==1 && niters > testcg_data.niters_max_prec)    testcg_data.niters_max_prec = niters;    // Same for preconditioned run
      if (A.geom->rank==0) {
        HPCG_fout << "Calling GMRES (all double) for testing: " << endl;
        HPCG_fout << "Call [" << i << "] Number of GMRES Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        HPCG_fout << " Expected " << expected_niters << " iterations.  Performed " << niters << "." << endl;
        HPCG_fout << " Time     " << time_solve << " seconds." << endl;
        HPCG_fout << " Gflop/s  " << flops/1000000000.0 << "/" << time_solve << " = " << (flops/1000000000.0)/time_solve 
                  << " (n = " << A.totalNumberOfRows << ")" << endl;
        HPCG_fout << " Time/itr " << time_solve / niters << endl;
      }
    }
  }

#if 1
  for (int k=(test_noprecond ? 0 : 1); k<2; ++k)
  { // This loop tests both unpreconditioned and preconditioned runs
    int expected_niters = testcg_data.expected_niters_no_prec;
    if (k==1) expected_niters = testcg_data.expected_niters_prec;
    for (int i=0; i< numberOfCgCalls; ++i) {
      ZeroVector(x); // Zero out x
      double time_tic = mytimer();
      int ierr = GMRES_IR(A, A_lo, data, data_lo, b, x, restart_length, maxIters, tolerance, niters, normr, normr0, &times[0], &flops, k);
      double time_solve = mytimer() - time_tic;
      if (ierr) HPCG_fout << "Error in call to GMRES-IR: " << ierr << ".\n" << endl;
      if (niters <= expected_niters) {
        ++testcg_data.count_pass;
      } else {
        ++testcg_data.count_fail;
      }
      if (k==0 && niters > testcg_data.niters_max_no_prec) testcg_data.niters_max_no_prec = niters; // Keep track of largest iter count
      if (k==1 && niters > testcg_data.niters_max_prec)    testcg_data.niters_max_prec = niters;    // Same for preconditioned run
      if (A.geom->rank==0) {
        HPCG_fout << "Call [" << i << "] Number of GMRES-IR Iterations [" << niters <<"] Scaled Residual [" << normr/normr0 << "]" << endl;
        HPCG_fout << " Expected " << expected_niters << " iterations.  Performed " << niters << "." << endl;
        HPCG_fout << " Time     " << time_solve << " seconds." << endl;
        HPCG_fout << " Gflop/s  " << flops/1000000000.0 << "/" << time_solve << " = " << (flops/1000000000.0)/time_solve 
                  << " (n = " << A.totalNumberOfRows << ")" << endl;
        HPCG_fout << " Time/itr " << time_solve / niters << endl;
      }
    }
  }
#endif
  // Restore matrix diagonal and RHS
  ReplaceMatrixDiagonal(A, origDiagA);
  ReplaceMatrixDiagonal(A_lo, origDiagA2);//TODO again, probably funny casting here. 
  CopyVector(origB, b);
  // Delete vectors
  DeleteVector(origDiagA);
  DeleteVector(exaggeratedDiagA);
  DeleteVector(origDiagA2);
  DeleteVector(exagDiagA2);
  DeleteVector(origB);
  testcg_data.normr = normr;

  return 0;
}

template<class SparseMatrix_type, class CGData_type, class Vector_type, class TestCGData_type>
int TestGMRES(SparseMatrix_type & A, CGData_type & data, Vector_type & b, Vector_type & x, TestCGData_type & testcg_data,
              bool test_diagonal_exaggeration, bool test_noprecond) {
  return TestGMRES(A, A, data, data, b, x, testcg_data, test_diagonal_exaggeration, test_noprecond);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int TestGMRES< SparseMatrix<double>, CGData<double>, Vector<double>, TestCGData<double> >
  (SparseMatrix<double>&, CGData<double>&, Vector<double>&, Vector<double>&, TestCGData<double>&, bool, bool);

template
int TestGMRES< SparseMatrix<float>, CGData<float>, Vector<float>, TestCGData<float> >
  (SparseMatrix<float>&, CGData<float>&, Vector<float>&, Vector<float>&, TestCGData<float>&, bool, bool);



// uniform version
template
int TestGMRES< SparseMatrix<double>, SparseMatrix<double>, CGData<double>, CGData<double>, Vector<double>, TestCGData<double> >
  (SparseMatrix<double>&, SparseMatrix<double>&, CGData<double>&, CGData<double>&, Vector<double>&, Vector<double>&, TestCGData<double>&, bool, bool);

template
int TestGMRES< SparseMatrix<float>, SparseMatrix<float>, CGData<float>, CGData<float>, Vector<float>, TestCGData<float> >
  (SparseMatrix<float>&, SparseMatrix<float>&, CGData<float>&, CGData<float>&, Vector<float>&, Vector<float>&, TestCGData<float>&, bool, bool);

// mixed version
template
int TestGMRES< SparseMatrix<double>, SparseMatrix<float>, CGData<double>, CGData<float>, Vector<double>, TestCGData<double> >
  (SparseMatrix<double>&, SparseMatrix<float>&, CGData<double>&, CGData<float>&, Vector<double>&, Vector<double>&, TestCGData<double>&, bool, bool);

