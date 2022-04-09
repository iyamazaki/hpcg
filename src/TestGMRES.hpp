
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
 @file TestGMRES.hpp

 HPGMRES data structure
 */

#ifndef TESTGMRES_HPP
#define TESTGMRES_HPP

#include "Hpgmp_Params.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"

template<class SC>
class TestCGData {
public:
  int count_pass; //!< number of succesful tests
  int count_fail;  //!< number of succesful tests
  int expected_niters_no_prec; //!< expected number of test CG iterations without preconditioning with diagonally dominant matrix (~12)
  int expected_niters_prec; //!< expected number of test CG iterations with preconditioning and with diagonally dominant matrix (~1-2)
  int niters_max_no_prec; //!< maximum number of test CG iterations without predictitioner
  int niters_max_prec; //!< maximum number of test CG iterations without predictitioner
  SC normr; //!< residual norm achieved during test CG iterations
};

template<class SparseMatrix_type, class SparseMatrix_type2, class CGData_type, class CGData_type2, class Vector_type, class TestCGData_type>
extern int TestGMRES(SparseMatrix_type & A, SparseMatrix_type2 & A_lo, CGData_type & data, CGData_type2 & data_lo, Vector_type & b, Vector_type & x, TestCGData_type & testcg_data,
                     bool test_diagonal_exaggeration, bool test_noprecond);

template<class SparseMatrix_type, class CGData_type, class Vector_type, class TestCGData_type>
extern int TestGMRES(SparseMatrix_type & A, CGData_type & data, Vector_type & b, Vector_type & x, TestCGData_type & testcg_data,
                     bool test_diagonal_exaggeration, bool test_noprecond);

#endif  // TESTGMRES_HPP

