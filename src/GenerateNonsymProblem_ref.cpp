
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
 @file GenerateProblem_ref.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#if defined(HPCG_DEBUG) || defined(HPCG_DETAILED_DEBUG)
#include <fstream>
using std::endl;
#include "hpcg.hpp"
#endif
#include <cassert>
#include <cmath>

#include "GenerateNonsymProblem_ref.hpp"


/*!
  Reference version of GenerateProblem to generate the sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A      The known system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

template<class SparseMatrix_type, class Vector_type>
void GenerateNonsymProblem_ref(SparseMatrix_type & A, Vector_type * b, Vector_type * x, Vector_type * xexact, bool init_vect) {

  typedef typename SparseMatrix_type::scalar_type matrix_scalar_type;
  typedef typename       Vector_type::scalar_type vector_scalar_type;
  const matrix_scalar_type zero (0.0);
  const matrix_scalar_type one  (1.0);
  const vector_scalar_type two = one + one;

  // Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
  // below may result in global range values.
  global_int_t nx = A.geom->nx;
  global_int_t ny = A.geom->ny;
  global_int_t nz = A.geom->nz;
  global_int_t gnx = A.geom->gnx;
  global_int_t gny = A.geom->gny;
  global_int_t gnz = A.geom->gnz;
  global_int_t gix0 = A.geom->gix0;
  global_int_t giy0 = A.geom->giy0;
  global_int_t giz0 = A.geom->giz0;

  local_int_t localNumberOfRows = nx*ny*nz; // This is the size of our subblock
  // If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
  assert(localNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)
  local_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil

  global_int_t totalNumberOfRows = gnx*gny*gnz; // Total number of grid points in mesh
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  assert(totalNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)


  // Allocate arrays that are of length localNumberOfRows
  char * nonzerosInRow = new char[localNumberOfRows];
  global_int_t ** mtxIndG = new global_int_t*[localNumberOfRows];
  local_int_t  ** mtxIndL = new local_int_t*[localNumberOfRows];
  matrix_scalar_type ** matrixValues   = new matrix_scalar_type*[localNumberOfRows];
  matrix_scalar_type ** matrixDiagonal = new matrix_scalar_type*[localNumberOfRows];

  vector_scalar_type * bv = 0;
  vector_scalar_type * xv = 0;
  vector_scalar_type * xexactv = 0;
  if (init_vect) {
    InitializeVector(*b, localNumberOfRows);
    InitializeVector(*x, localNumberOfRows);
    InitializeVector(*xexact, localNumberOfRows);
    bv = b->values; // Only compute exact solution if requested
    xv = x->values; // Only compute exact solution if requested
    xexactv = xexact->values; // Only compute exact solution if requested
  }
  A.localToGlobalMap.resize(localNumberOfRows);

  // Use a parallel loop to do initial assignment:
  // distributes the physical placement of arrays of pointers across the memory system
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< localNumberOfRows; ++i) {
    matrixValues[i] = 0;
    matrixDiagonal[i] = 0;
    mtxIndG[i] = 0;
    mtxIndL[i] = 0;
  }

#ifndef HPCG_CONTIGUOUS_ARRAYS
  // Now allocate the arrays pointed to
  for (local_int_t i=0; i< localNumberOfRows; ++i)
    mtxIndL[i] = new local_int_t[numberOfNonzerosPerRow];
  for (local_int_t i=0; i< localNumberOfRows; ++i)
    matrixValues[i] = new matrix_scalar_type[numberOfNonzerosPerRow];
  for (local_int_t i=0; i< localNumberOfRows; ++i)
   mtxIndG[i] = new global_int_t[numberOfNonzerosPerRow];

#else
  // Now allocate the arrays pointed to
  mtxIndL[0] = new local_int_t[localNumberOfRows * numberOfNonzerosPerRow];
  matrixValues[0] = new matrix_scalar_type[localNumberOfRows * numberOfNonzerosPerRow];
  mtxIndG[0] = new global_int_t[localNumberOfRows * numberOfNonzerosPerRow];

  for (local_int_t i=1; i< localNumberOfRows; ++i) {
    mtxIndL[i] = mtxIndL[0] + i * numberOfNonzerosPerRow;
    matrixValues[i] = matrixValues[0] + i * numberOfNonzerosPerRow;
    mtxIndG[i] = mtxIndG[0] + i * numberOfNonzerosPerRow;
  }
#endif

  matrix_scalar_type beta (1.0);
  matrix_scalar_type gamma (10.0); //one;
  local_int_t localNumberOfNonzeros = 0;
  // TODO:  This triply nested loop could be flattened or use nested parallelism
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t iz=0; iz<nz; iz++) {
    global_int_t giz = giz0+iz;
    for (local_int_t iy=0; iy<ny; iy++) {
      global_int_t giy = giy0+iy;
      for (local_int_t ix=0; ix<nx; ix++) {
        global_int_t gix = gix0+ix;
        local_int_t currentLocalRow = iz*(nx*ny) + iy*(nx) + ix;
        global_int_t currentGlobalRow = giz*(gnx*gny) + giy*(gnx) + gix;
#ifndef HPCG_NO_OPENMP
// C++ std::map is not threadsafe for writing
        #pragma omp critical
#endif
        A.globalToLocalMap[currentGlobalRow] = currentLocalRow;

        A.localToGlobalMap[currentLocalRow] = currentGlobalRow;
#ifdef HPCG_DETAILED_DEBUG
        HPCG_fout << " rank, globalRow, localRow = " << A.geom->rank << " " << currentGlobalRow << " " << A.globalToLocalMap[currentGlobalRow] << endl;
#endif
        char numberOfNonzerosInRow = 0;
        matrix_scalar_type * currentValuePointer = matrixValues[currentLocalRow]; // Pointer to current value in current row
        vector_scalar_type bi (0.0);
        global_int_t * currentIndexPointerG = mtxIndG[currentLocalRow]; // Pointer to current index in current row
        for (int sz=-1; sz<=1; sz++) {
          int jz = iz+sz;
          if (giz+sz>-1 && giz+sz<gnz) {
            for (int sy=-1; sy<=1; sy++) {
              int jy = iy+sy;
              if (giy+sy>-1 && giy+sy<gny) {
                for (int sx=-1; sx<=1; sx++) {
                  int jx = ix+sx;
                  if (gix+sx>-1 && gix+sx<gnx) {
                    global_int_t curcol = currentGlobalRow + sz*(gnx*gny) + sy*(gnx) + sx;
                    if (curcol==currentGlobalRow) {
                      matrixDiagonal[currentLocalRow] = currentValuePointer;
                      *currentValuePointer = 26.0;
                    } else {
                      *currentValuePointer = 1.0;
                    }
                    matrix_scalar_type beta_i = sqrt(one + beta*(((matrix_scalar_type)(gix0+ix))/((matrix_scalar_type)(gnx-1)))) *
                                                sqrt(one + beta*(((matrix_scalar_type)(giy0+iy))/((matrix_scalar_type)(gny-1)))) *
                                                sqrt(one + beta*(((matrix_scalar_type)(giz0+iz))/((matrix_scalar_type)(gnz-1))));
                    matrix_scalar_type beta_j = sqrt(one + beta*(((matrix_scalar_type)(gix0+jx))/((matrix_scalar_type)(gnx-1)))) *
                                                sqrt(one + beta*(((matrix_scalar_type)(giy0+jy))/((matrix_scalar_type)(gny-1)))) *
                                                sqrt(one + beta*(((matrix_scalar_type)(giz0+jz))/((matrix_scalar_type)(gnz-1))));
                    *currentValuePointer *= (beta_i * beta_j);
                    if (sy == 0 && sz == 0) {
                      if (sx == 1) {
                        *currentValuePointer += (gamma / two);
                      } else if (sx == -1) {
                        *currentValuePointer -= (gamma / two);
                      }
                    }
                    bi += *currentValuePointer ;
                    *currentIndexPointerG++ = curcol;
                    *currentValuePointer++;
                    numberOfNonzerosInRow++;
                  } // end x bounds test
                } // end sx loop
              } // end y bounds test
            } // end sy loop
          } // end z bounds test
        } // end sz loop
        nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
#ifndef HPCG_NO_OPENMP
        #pragma omp critical
#endif
        localNumberOfNonzeros += numberOfNonzerosInRow; // Protect this with an atomic
        if (init_vect) {
          bv[currentLocalRow] = bi; //26.0 - ((double) (numberOfNonzerosInRow-1));
          xv[currentLocalRow] = zero;
          xexactv[currentLocalRow] = one;
        }
      } // end ix loop
    } // end iy loop
  } // end iz loop
#ifdef HPCG_DETAILED_DEBUG
  HPCG_fout     << "Process " << A.geom->rank << " of " << A.geom->size <<" has " << localNumberOfRows    << " rows."     << endl
      << "Process " << A.geom->rank << " of " << A.geom->size <<" has " << localNumberOfNonzeros<< " nonzeros." <<endl;
#endif

  global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
  MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
  long long lnnz = localNumberOfNonzeros, gnnz = 0; // convert to 64 bit for MPI call
  MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
  totalNumberOfNonzeros = gnnz; // Copy back
#endif
#else
  totalNumberOfNonzeros = localNumberOfNonzeros;
#endif
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  // This assert is usually the first to fail as problem size increases beyond the 32-bit integer range.
  assert(totalNumberOfNonzeros>0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

  A.title = 0;
  A.totalNumberOfRows = totalNumberOfRows;
  A.totalNumberOfNonzeros = totalNumberOfNonzeros;
  A.localNumberOfRows = localNumberOfRows;
  A.localNumberOfColumns = localNumberOfRows;
  A.localNumberOfNonzeros = localNumberOfNonzeros;
  A.nonzerosInRow = nonzerosInRow;
  A.mtxIndG = mtxIndG;
  A.mtxIndL = mtxIndL;
  A.matrixValues = matrixValues;
  A.matrixDiagonal = matrixDiagonal;

  return;
}


/* --------------- *
 * specializations *
 * --------------- */

// uniform
template
void GenerateNonsymProblem_ref< SparseMatrix<double>, Vector<double> >(SparseMatrix<double>&, Vector<double>*, Vector<double>*, Vector<double>*, bool);

template
void GenerateNonsymProblem_ref< SparseMatrix<float>, Vector<float> >(SparseMatrix<float>&, Vector<float>*, Vector<float>*, Vector<float>*, bool);


// mixed
template
void GenerateNonsymProblem_ref< SparseMatrix<float>, Vector<double> >(SparseMatrix<float>&, Vector<double>*, Vector<double>*, Vector<double>*, bool);

