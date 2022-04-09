
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
 @file TestSymmetry.hpp

 HPCG data structures for symmetry testing
 */

#ifndef TESTSYMMETRY_HPP
#define TESTSYMMETRY_HPP

#include "Hpgmp_Params.hpp"
#include "SparseMatrix.hpp"
#include "CGData.hpp"

template<class SC>
class TestSymmetryData {
public:
  SC     depsym_spmv;  //!< departure from symmetry for the SPMV kernel
  SC     depsym_mg; //!< departure from symmetry for the MG kernel
  int    count_fail;   //!< number of failures in the symmetry tests
};

template<class SparseMatrix_type, class Vector_type, class TestSymmetryData_type>
extern int TestSymmetry(SparseMatrix_type & A, Vector_type & b, Vector_type & xexact, TestSymmetryData_type & testsymmetry_data);

#endif  // TESTSYMMETRY_HPP
