
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
 @file MGData.hpp

 HPCG data structure
 */

#ifndef MGDATA_HPP
#define MGDATA_HPP

#include <cassert>
#include "SparseMatrix.hpp"
#include "Vector.hpp"

#ifdef HPCG_WITH_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

template<class SC>
class MGData {
public:
  typedef Vector<SC> Vector_type;
  int numberOfPresmootherSteps; // Call ComputeSYMGS this many times prior to coarsening
  int numberOfPostsmootherSteps; // Call ComputeSYMGS this many times after coarsening
  local_int_t * f2cOperator; //!< 1D array containing the fine operator local IDs that will be injected into coarse space.
  Vector_type * rc; // coarse grid residual vector
  Vector_type * xc; // coarse grid solution vector
  Vector_type * Axf; // fine grid residual vector
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;
  #ifdef HPCG_WITH_CUDA
  // to store the restrictiion as CRS matrix on device
  cusparseMatDescr_t descrA;
  int *d_row_ptr;
  int *d_col_idx;
  SC  *d_nzvals;   //!< values of matrix entries
  #endif
};

/*!
 Constructor for the data structure of CG vectors.

 @param[in] Ac - Fully-formed coarse matrix
 @param[in] f2cOperator -
 @param[out] data the data structure for CG vectors that will be allocated to get it ready for use in CG iterations
 */
template <class MGData_type>
inline void InitializeMGData(local_int_t * f2cOperator,
                             typename MGData_type::Vector_type * rc,
                             typename MGData_type::Vector_type * xc,
                             typename MGData_type::Vector_type * Axf,
                             MGData_type & data) {
  data.numberOfPresmootherSteps = 1;
  data.numberOfPostsmootherSteps = 1;
  data.f2cOperator = f2cOperator; // Space for injection operator
  data.rc = rc;
  data.xc = xc;
  data.Axf = Axf;
  return;
}

/*!
 Destructor for the CG vectors data.

 @param[inout] data the MG data structure whose storage is deallocated
 */
template <class MGData_type>
inline void DeleteMGData(MGData_type & data) {

  delete [] data.f2cOperator;
  DeleteVector(*data.Axf);
  DeleteVector(*data.rc);
  DeleteVector(*data.xc);
  delete data.Axf;
  delete data.rc;
  delete data.xc;
  return;
}

#endif // MGDATA_HPP

