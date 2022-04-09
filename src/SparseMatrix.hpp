
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
 @file SparseMatrix.hpp

 HPCG data structures for the sparse matrix
 */

#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <vector>
#include <cassert>
#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"
#if __cplusplus < 201103L
// for C++03
#include <map>
typedef std::map< global_int_t, local_int_t > GlobalToLocalMap;
#else
// for C++11 or greater
#include <unordered_map>
using GlobalToLocalMap = std::unordered_map< global_int_t, local_int_t >;
#endif

#ifdef HPCG_WITH_CUDA
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

template <class SC = double>
class SparseMatrix {
public:
  typedef SC scalar_type;
  char  * title; //!< name of the sparse matrix
  Geometry * geom; //!< geometry associated with this matrix
  global_int_t totalNumberOfRows; //!< total number of matrix rows across all processes
  global_int_t totalNumberOfNonzeros; //!< total number of matrix nonzeros across all processes
  global_int_t totalNumberOfMGNonzeros; //!< total number of matrix nonzeros across all processes, for MG
  local_int_t localNumberOfRows; //!< number of rows local to this process
  local_int_t localNumberOfColumns;  //!< number of columns local to this process
  local_int_t localNumberOfNonzeros;  //!< number of nonzeros local to this process
  local_int_t localNumberOfMGNonzeros;  //!< number of nonzeros local to this process, for MG
  char  * nonzerosInRow;  //!< The number of nonzeros in a row will always be 27 or fewer
  global_int_t ** mtxIndG; //!< matrix indices as global values
  local_int_t ** mtxIndL; //!< matrix indices as local values
  SC ** matrixValues; //!< values of matrix entries
  SC ** matrixDiagonal; //!< values of matrix diagonal entries
  GlobalToLocalMap globalToLocalMap; //!< global-to-local mapping
  std::vector< global_int_t > localToGlobalMap; //!< local-to-global mapping
  mutable bool isDotProductOptimized;
  mutable bool isSpmvOptimized;
  mutable bool isMgOptimized;
  mutable bool isWaxpbyOptimized;
  mutable bool isGemvOptimized;
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  mutable SparseMatrix<SC> * Ac;   // Coarse grid matrix
  mutable MGData<SC> * mgData; // Pointer to the coarse level data for this fine matrix
  void * optimizationData;  // pointer that can be used to store implementation-specific data

#ifndef HPCG_NO_MPI
  local_int_t numberOfExternalValues; //!< number of entries that are external to this process
  int numberOfSendNeighbors; //!< number of neighboring processes that will be send local data
  local_int_t totalToBeSent; //!< total number of entries to be sent
  local_int_t * elementsToSend; //!< elements to send to neighboring processes
  int * neighbors; //!< neighboring processes
  local_int_t * receiveLength; //!< lenghts of messages received from neighboring processes
  local_int_t * sendLength; //!< lenghts of messages sent to neighboring processes
  SC * sendBuffer; //!< send buffer for non-blocking sends
#endif
#if defined(HPCG_WITH_CUDA) | defined(HPCG_WITH_HIP)
  #if defined(HPCG_WITH_CUDA)
  cusparseHandle_t cusparseHandle;
  cusparseMatDescr_t descrA;
  #endif

  // to store the local matrix on device
  int *d_row_ptr;
  int *d_col_idx;
  SC  *d_nzvals;   //!< values of matrix entries

  // to store the lower-triangular matrix on device
  local_int_t nnzL;
  #if defined(HPCG_WITH_CUDA)
  cusparseMatDescr_t descrL;
  cusparseSolveAnalysisInfo_t infoL;
  #endif
  int *d_Lrow_ptr;
  int *d_Lcol_idx;
  SC  *d_Lnzvals;   //!< values of matrix entries
  // to store the strictly upper-triangular matrix on device
  local_int_t nnzU;
  #if defined(HPCG_WITH_CUDA)
  cusparseMatDescr_t descrU;
  #endif
  int *d_Urow_ptr;
  int *d_Ucol_idx;
  SC  *d_Unzvals;   //!< values of matrix entries

  // TODO: remove
  Vector<SC> x; // nrow
  Vector<SC> y; // ncol
#endif
};

/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
template<class SparseMatrix_type>
inline void InitializeSparseMatrix(SparseMatrix_type & A, Geometry * geom) {
  A.title = 0;
  A.geom = geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.totalNumberOfMGNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  A.localNumberOfMGNonzeros = 0;
  A.nonzerosInRow = 0;
  A.mtxIndG = 0;
  A.mtxIndL = 0;
  A.matrixValues = 0;
  A.matrixDiagonal = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized       = true;
  A.isMgOptimized         = true;
  A.isWaxpbyOptimized     = true;
  A.isGemvOptimized       = true;

#ifndef HPCG_NO_MPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  A.elementsToSend = 0;
  A.neighbors = 0;
  A.receiveLength = 0;
  A.sendLength = 0;
  A.sendBuffer = 0;
#endif
  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac =0;
  return;
}

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
template <class SparseMatrix_type, class Vector_type>
inline void CopyMatrixDiagonal(SparseMatrix_type & A, Vector_type & diagonal) {
  typedef typename SparseMatrix_type::scalar_type scalar_type;
  scalar_type ** curDiagA = A.matrixDiagonal;
  scalar_type * dv = diagonal.values;
  assert(A.localNumberOfRows==diagonal.localLength);
  for (local_int_t i=0; i<A.localNumberOfRows; ++i) dv[i] = *(curDiagA[i]);
  return;
}
/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
template <class SparseMatrix_type, class Vector_type>
inline void ReplaceMatrixDiagonal(SparseMatrix_type & A, Vector_type & diagonal) {
  typedef typename SparseMatrix_type::scalar_type scalar_type;
  scalar_type ** curDiagA = A.matrixDiagonal;
  scalar_type * dv = diagonal.values;
  assert(A.localNumberOfRows==diagonal.localLength);
  for (local_int_t i=0; i<A.localNumberOfRows; ++i) *(curDiagA[i]) = dv[i];
  return;
}
/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
template <class SparseMatrix_type>
inline void DeleteMatrix(SparseMatrix_type & A) {

#ifndef HPCG_CONTIGUOUS_ARRAYS
  for (local_int_t i = 0; i< A.localNumberOfRows; ++i) {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndG[i];
    delete [] A.mtxIndL[i];
  }
#else
  delete [] A.matrixValues[0];
  delete [] A.mtxIndG[0];
  delete [] A.mtxIndL[0];
#endif
  if (A.title)                 delete [] A.title;
  if (A.nonzerosInRow)         delete [] A.nonzerosInRow;
  if (A.mtxIndG)               delete [] A.mtxIndG;
  if (A.mtxIndL)               delete [] A.mtxIndL;
  if (A.matrixValues)          delete [] A.matrixValues;
  if (A.matrixDiagonal)        delete [] A.matrixDiagonal;

#ifndef HPCG_NO_MPI
  if (A.elementsToSend)        delete [] A.elementsToSend;
  if (A.neighbors)             delete [] A.neighbors;
  if (A.receiveLength)         delete [] A.receiveLength;
  if (A.sendLength)            delete [] A.sendLength;
  if (A.sendBuffer)            delete [] A.sendBuffer;
#endif

  if (A.geom!=0) {
    DeleteGeometry(*A.geom);
    delete A.geom;
    A.geom = 0;
  }
  if (A.Ac!=0) {
    // Delete coarse matrix
    DeleteMatrix(*A.Ac);
    delete A.Ac; 
    A.Ac = 0;
  }
  if (A.mgData!=0) {
    // Delete MG data
    DeleteMGData(*A.mgData);
    delete A.mgData;
    A.mgData = 0;
  }

#ifdef HPCG_WITH_CUDA
  cudaFree (A.d_row_ptr);
  cudaFree (A.d_col_idx);
  cudaFree (A.d_nzvals);

  cudaFree (A.d_Lrow_ptr);
  cudaFree (A.d_Lcol_idx);
  cudaFree (A.d_Lnzvals);

  cudaFree (A.d_Urow_ptr);
  cudaFree (A.d_Ucol_idx);
  cudaFree (A.d_Unzvals);

  DeleteVector (A.x);
  DeleteVector (A.y);

  cusparseDestroy(A.cusparseHandle);
  cusparseDestroyMatDescr(A.descrA);
  cusparseDestroyMatDescr(A.descrL);
  cusparseDestroyMatDescr(A.descrU);
  cusparseDestroySolveAnalysisInfo(A.infoL);
#endif
  return;
}

#endif // SPARSEMATRIX_HPP
