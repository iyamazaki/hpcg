
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

 HPGMP routine
 */

// Main routine of a program that calls the HPGMP GMRES and GMRES-IR 
// solvers to solve the problem, and then prints results.

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
#include "SetupHalo.hpp"
#include "CheckProblem.hpp"
#include "ExchangeHalo.hpp"
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

#include "GMRES.hpp"
#include "TestGMRES.hpp"

typedef double scalar_type;
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

  //Initialize params for full-scale benchmark run:
  HPCG_Params params;
  HPCG_Init(&argc, &argv, params);
  int size = params.comm_size; // Num MPI processes
  int rank = params.comm_rank; // My process ID

// **************************************************************************8
// PHASE I: VERIFY CONVERGENCE WITH SMALL PROBLEM
// **************************************************************************8

  MPI_Comm SM_COMM;
  MPI_Group group_world;
  MPI_Group sm_group;
  if( size <= 64 ) //Use all MPI ranks
    SM_COMM = MPI_COMM_WORLD;
  else{
    // Get MPI Sub-communicator:
    int num_ranks = 64;
    int *process_ranks;
    // make a list of processes in the new communicator
    process_ranks = (int*) malloc(num_ranks*sizeof(int));
    for(int i = 0; i < num_ranks; i++)
      process_ranks[i] = i;
    //get the group under MPI_COMM_WORLD
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    // create the new group
    MPI_Group_incl(group_world, num_ranks, process_ranks, &sm_group);
    // create the new communicator
    MPI_Comm_create(MPI_COMM_WORLD, sm_group, &SM_COMM);
  }

  HPCG_Params sm_params;
  sm_params.nx = 32;
  sm_params.ny = 32;
  sm_params.nz = 32;

  sm_params.runningTime = 0.0005; //Something really small since we just want one run??
  sm_params.pz = 0;
  sm_params.zl = 0;
  sm_params.zu = 0;

  sm_params.npx = 0;
  sm_params.npy = 0;
  sm_params.npz = 0;

#ifndef HPCG_NO_MPI
  MPI_Comm_rank( SM_COMM, &sm_params.comm_rank );
  MPI_Comm_size( SM_COMM, &sm_params.comm_size );
#else
  sm_params.comm_rank = 0;
  sm_params.comm_size = 1;
#endif

#ifdef HPCG_NO_OPENMP
  sm_params.numThreads = 1;
#else
  #pragma omp parallel
  sm_params.numThreads = omp_get_num_threads();
#endif

  // Construct the geometry and linear system
  local_int_t nx,ny,nz;
  sm_nx = (local_int_t)sm_params.nx;
  sm_ny = (local_int_t)sm_params.ny;
  sm_nz = (local_int_t)sm_params.nz;
  Geometry * sm_geom = new Geometry;
  GenerateGeometry(sm_params.comm_size, sm_params.comm_rank, sm_params.numThreads, sm_params.pz, 
      sm_params.zl, sm_params.zu, sm_nx, sm_ny, sm_nz, sm_params.npx, sm_params.npy, sm_params.npz, sm_geom);

// **************************************************************************8
// PHASE II: BENCHMARKING PHASE
// **************************************************************************8

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  bool quickPath = 1; //TODO: Change back to the following after=(params.runningTime==0);

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
  CGData_type data;//TODO What is this for?

  bool init_vect = true;
  Vector_type b, x, xexact;

  int numberOfMgLevels = 4; // Number of levels including first
  SetupProblem(numberOfMgLevels, A, geom, data, &b, &x, &xexact, init_vect);

  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  //TODO: This is the spot where HPCG runs check problem.  Do we need CheckProblem?
  //Probably need to check multigird (and Traingular solve?) here. 

  // Call user-tunable set up function.
  double t7 = mytimer();
  OptimizeProblem(A, data, b, x, xexact);
  t7 = mytimer() - t7;
  times[7] = t7;

  if (A.geom->rank==0) {
    HPCG_fout << " Setup    Time     " << setup_time << " seconds." << endl;
    HPCG_fout << " Optimize Time     " << t7 << " seconds." << endl;
  }

  ///////////////////////////////
  // Reference GMRES Timing Phase //
  ///////////////////////////////

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  int global_failure = 0; // assume all is well: no failures

  int niters = 0;
  int totalNiters_ref = 0;
  scalar_type normr = 0.0;
  scalar_type normr0 = 0.0;
  int restart_length = 50;
  int refMaxIters = 50;

  // Compute the residual reduction for the natural ordering and reference kernels
  double flops = 0.0;
  std::vector< double > ref_times(9,0.0);
  scalar_type tolerance = 0.0; // Set tolerance to zero to make all runs do maxIters iterations
  int err_count = 0;
  ZeroVector(x);
  ierr = GMRES(A, data, b, x, restart_length, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], &flops, true);
  if (ierr) ++err_count; // count the number of errors in GMRES.
  totalNiters_ref += niters;
  if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to reference GMRES." << endl;
  scalar_type refTolerance = normr / normr0;

  // Call user-tunable set up function.
  t7 = mytimer();
  OptimizeProblem(A, data, b, x, xexact);
  t7 = mytimer() - t7;
  times[7] = t7;
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
#endif

#ifdef HPCG_DETAILED_DEBUG
  if (geom->size == 1) WriteProblem(*geom, A, b, x, xexact);
#endif


  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////
  TestCGData_type testcg_data;
  //
//Don't need this anymore.  both double and MP GMRES currently tested in next call. 
/*
#ifdef HPCG_DEBUG
  t1 = mytimer();
  if (rank==0) HPCG_fout << endl << "Running Uniform-precision Test" << endl;
#endif
  testcg_data.count_pass = testcg_data.count_fail = 0;
  TestGMRES(A, data, b, x, testcg_data);

  //TODO: Replace with a test of non-symm MG Gauss-Seidel smooother.  Pass data as arg to ReportResults.
  //TestSymmetryData_type testsymmetry_data;
  //TestSymmetry(A, b, xexact, testsymmetry_data);

#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total validation (TestGMRES) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
*/ 

/////////////////////////////////////////
// Mixed precision test phase from Ichi:
// ////////////////////////////////////
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

  bool test_diagonal_exaggeration = true;
  bool test_noprecond = true;
#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  TestGMRES(A, A2, data, data2, b, x, testcg_data, test_diagonal_exaggeration, test_noprecond);
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total validation (mixed-precision TestGMRES) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  //////////////////////////////
  // Optimized CG Setup Phase //
  //////////////////////////////
  int optMaxIters = 10*refMaxIters;
  ///////////////////////////////
  // Optimized CG Timing Phase //
  ///////////////////////////////
  int numberOfCgSets = 1; //TODO change this. 
  
  TestNormsData_type testnorms_data;

  ////////////////////
  // Report Results //
  ////////////////////

  // Report results to YAML file
  ReportResults(A, numberOfMgLevels, numberOfCgSets, refMaxIters, optMaxIters, &times[0], testcg_data, testnorms_data, global_failure, quickPath);

  // Clean up
  DeleteMatrix(A); // This delete will recursively delete all coarse grid data
  DeleteCGData(data);
  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);
  //delete [] testnorms_data.values;

  // Finish up
  HPCG_Finalize();
#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
