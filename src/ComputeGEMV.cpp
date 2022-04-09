
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
 @file Vector.hpp

 HPCG data structures for dense vectors
 */

#include "ComputeGEMV.hpp"
#include "ComputeGEMV_ref.hpp"

template<class MultiVector_type, class Vector_type, class SerialDenseMatrix_type>
int ComputeGEMV(const local_int_t m, const local_int_t n,
                const typename MultiVector_type::scalar_type alpha, const MultiVector_type & A, const SerialDenseMatrix_type & x,
                const typename      Vector_type::scalar_type beta,  const Vector_type & y,
                bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeGEMV should be used.
  isOptimized = false;
  return ComputeGEMV_ref(m, n, alpha, A, x, beta, y);
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
int ComputeGEMV< MultiVector<double>, Vector<double>, SerialDenseMatrix<double> >
  (int, int, double, MultiVector<double> const&, SerialDenseMatrix<double> const&, double, Vector<double> const&, bool&);

template
int ComputeGEMV< MultiVector<float>, Vector<float>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, float, Vector<float> const&, bool&);


// mixed
template
int ComputeGEMV< MultiVector<float>, Vector<double>, SerialDenseMatrix<float> >
  (int, int, float, MultiVector<float> const&, SerialDenseMatrix<float> const&, double, Vector<double> const&, bool&);

