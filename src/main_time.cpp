
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
 @file main.cpp

 HPCG routine
 */

// Main routine of a program that calls the HPCG conjugate gradient
// solver to solve the problem, and then prints results.

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>

#include "Hpgmp_Params.hpp"

#include "SetupProblem.hpp"
#include "CheckAspectRatio.hpp"
#include "GenerateGeometry.hpp"
#include "CheckProblem.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestNorms.hpp"

#include "TestGMRES.hpp"

typedef double scalar_type;
//typedef float  scalar_type;
typedef Vector<scalar_type> Vector_type;
typedef SparseMatrix<scalar_type> SparseMatrix_type;
typedef CGData<scalar_type> CGData_type;
typedef TestCGData<scalar_type> TestCGData_type;
typedef TestNormsData<scalar_type> TestNormsData_type;

typedef float scalar_type2;
typedef Vector<scalar_type2> Vector_type2;
typedef SparseMatrix<scalar_type2> SparseMatrix_type2;
typedef CGData<scalar_type2> CGData_type2;
typedef TestCGData<scalar_type2> TestCGData_type2;
typedef TestNormsData<scalar_type2> TestNormsData_type2;


/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char * argv[]) {

#ifndef HPCG_NO_MPI
  MPI_Init(&argc, &argv);
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  bool quickPath = (params.runningTime==0);

  int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

#ifdef HPCG_DETAILED_DEBUG
  if (size < 100 && rank==0) HPCG_fout << "Process "<<rank<<" of "<<size<<" is alive with " << params.numThreads << " threads." <<endl;

  if (rank==0) {
    char c;
    std::cout << "Press key to continue"<< std::endl;
    std::cin.get(c);
  }
#ifndef HPCG_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

  local_int_t nx,ny,nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;  // Used to check return codes on function calls

  ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank==0);
  if (ierr)
    return ierr;

  /////////////////////////
  // Problem setup Phase //
  /////////////////////////

#ifdef HPCG_DEBUG
  double t1 = mytimer();
#endif

  // Construct the geometry and linear system
  Geometry * geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);

  ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank==0);
  if (ierr)
    return ierr;

  // Use this array for collecting timing information
  std::vector< double > times(10,0.0);

  double setup_time = mytimer();

  // Setup the problem
  SparseMatrix_type A;
  CGData_type data;

  bool init_vect = true;
  Vector_type b, x, xexact;

  int numberOfMgLevels = 4; // Number of levels including first
  SetupProblem(numberOfMgLevels, A, geom, data, &b, &x, &xexact, init_vect);

  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  // Call user-tunable set up function.
  double t7 = mytimer();
  OptimizeProblem(A, data, b, x, xexact);
  t7 = mytimer() - t7;
  times[7] = t7;

  if (A.geom->rank==0) {
    HPCG_fout << " Setup    Time     " << setup_time << " seconds." << endl;
    HPCG_fout << " Optimize Time     " << t7 << " seconds." << endl;
  }

  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

  // Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines

  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector_type x_overlap, b_computed;
  InitializeVector(x_overlap, ncol); // Overlapped copy of x vector
  InitializeVector(b_computed, nrow); // Computed RHS vector


  // Record execution time of reference SpMV and MG kernels for reporting times
  // First load vector with random values
  FillRandomVector(x_overlap);

#if 0
  int numberOfCalls = 10;
  if (quickPath) numberOfCalls = 1; //QuickPath means we do on one call of each block of repetitive code
  double t_begin = mytimer();
  for (int i=0; i< numberOfCalls; ++i) {
    ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
    if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap
    if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  }
  times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total SpMV+MG timing phase execution time in main (sec) = " << mytimer() - t1 << endl;
#endif
#endif


  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////

  TestCGData_type testcg_data;
  testcg_data.count_pass = testcg_data.count_fail = 0;

  bool test_diagonal_exaggeration = false;
  bool test_noprecond = false;

#ifdef HPCG_DEBUG
  t1 = mytimer();
  if (rank==0) HPCG_fout << endl << "Running Uniform-precision Test" << endl;
#endif
  TestGMRES(A, data, b, x, testcg_data, test_diagonal_exaggeration, test_noprecond);
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total validation (uniform-precision TestGMRES) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

  setup_time = mytimer();
  init_vect = false;
  SparseMatrix_type2 A2;
  CGData_type2 data2;
  SetupProblem(numberOfMgLevels, A2, geom, data2, &b, &x, &xexact, init_vect);
  setup_time = mytimer() - setup_time; // Capture total time of setup

  t7 = mytimer();
  OptimizeProblem(A2, data, b, x, xexact);
  t7 = mytimer() - t7;

  testcg_data.count_pass = testcg_data.count_fail = 0;
  if (A.geom->rank==0) {
    HPCG_fout << " Setup    Time     " << setup_time << " seconds." << endl;
    HPCG_fout << " Optimize Time     " << t7 << " seconds." << endl;
  }


#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  TestGMRES(A, A2, data, data2, b, x, testcg_data, test_diagonal_exaggeration, test_noprecond);
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total validation (mixed-precision TestGMRES) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

  // free
  //DeleteMatrix(A2);
  //DeleteMatrix(A);
  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);
  DeleteVector(x_overlap);
  DeleteVector(b_computed);
  DeleteCGData(data);
  DeleteCGData(data2);

  // Finish up
  HPCG_Finalize();
#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
