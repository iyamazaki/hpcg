
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
 @file Hpgmp_Params.hpp

 HPCG data structures and functions
 */

#ifndef HPGMP_PARAMS_HPP
#define HPGMP_PARAMS_HPP

#include <fstream>
#include "Geometry.hpp"

extern std::ofstream HPCG_fout;

struct HPCG_Params_STRUCT {
  int comm_size; //!< Number of MPI processes in MPI_COMM_WORLD
  int comm_rank; //!< This process' MPI rank in the range [0 to comm_size - 1]
  int numThreads; //!< This process' number of threads
  local_int_t nx; //!< Number of x-direction grid points for each local subdomain
  local_int_t ny; //!< Number of y-direction grid points for each local subdomain
  local_int_t nz; //!< Number of z-direction grid points for each local subdomain
  int runningTime; //!< Number of seconds to run the timed portion of the benchmark
  int npx; //!< Number of processes in z-direction of 3D process grid
  int npy; //!< Number of processes in z-direction of 3D process grid
  int npz; //!< Number of processes in z-direction of 3D process grid
  int pz; //!< Partition in the z processor dimension, default is npz
  local_int_t zl; //!< nz for processors in the z dimension with value less than pz
  local_int_t zu; //!< nz for processors in the z dimension with value greater than pz
};
/*!
  HPCG_Params is a shorthand for HPCG_Params_STRUCT
 */
typedef HPCG_Params_STRUCT HPCG_Params;

extern int HPCG_Init(int * argc_p, char ** *argv_p, HPCG_Params & params);
extern int HPCG_Finalize(void);

#endif // HPCG_HPP
