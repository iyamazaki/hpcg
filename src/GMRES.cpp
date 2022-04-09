
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
 @file GMRES.cpp

 GMRES routine
 */

#include <fstream>

#include <cmath>

#include "Hpgmp_Params.hpp"

#include "GMRES.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"
#include "ComputeTRSM.hpp"
#include "ComputeGEMV.hpp"
#include "ComputeGEMVT.hpp"


// Use TICK and TOCK to time a code section in MATLAB-like fashion
#define TICK()  t0 = mytimer() //!< record current time in 't0'
#define TOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see GMRES_ref()
*/
template<class SparseMatrix_type, class CGData_type, class Vector_type>
int GMRES(const SparseMatrix_type & A, CGData_type & data, const Vector_type & b, Vector_type & x,
          const int restart_length, const int max_iter, const typename SparseMatrix_type::scalar_type tolerance,
          int & niters, typename SparseMatrix_type::scalar_type & normr,  typename SparseMatrix_type::scalar_type & normr0,
          double * times, double * flops, bool doPreconditioning) {
 
  typedef typename SparseMatrix_type::scalar_type scalar_type;
  typedef MultiVector<scalar_type> MultiVector_type;
  typedef SerialDenseMatrix<scalar_type> SerialDenseMatrix_type;

  const scalar_type one  (1.0);
  const scalar_type zero (0.0);
  double t_begin = mytimer();  // Start timing right away
  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0, t6 = 0.0;

  normr = 0.0;
  scalar_type rtz = zero, oldrtz = zero, alpha = zero, beta = zero, pAp = zero;

//#ifndef HPCG_NO_MPI
//  double t6 = 0.0;
//#endif
  local_int_t nrow = A.localNumberOfRows;
  local_int_t Nrow = A.totalNumberOfRows;
  Vector_type & r = data.r; // Residual vector
  Vector_type & z = data.z; // Preconditioned residual vector
  Vector_type & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector_type & Ap = data.Ap;

  SerialDenseMatrix_type H;
  SerialDenseMatrix_type h;
  SerialDenseMatrix_type t;
  SerialDenseMatrix_type cs;
  SerialDenseMatrix_type ss;
  MultiVector_type Q;
  MultiVector_type P;
  Vector_type Qkm1;
  Vector_type Qk;
  Vector_type Qj;
  InitializeMatrix(H,  restart_length+1, restart_length);
  InitializeMatrix(h,  restart_length+1, 1);
  InitializeMatrix(t,  restart_length+1, 1);
  InitializeMatrix(cs, restart_length+1, 1);
  InitializeMatrix(ss, restart_length+1, 1);
  InitializeMultiVector(Q, nrow, restart_length+1);

  if (!doPreconditioning && A.geom->rank==0) HPCG_fout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

  bool verbose = true;
  int print_freq = 1;
  if (verbose && A.geom->rank==0) {
    HPCG_fout << std::endl << " Running GMRES(" << restart_length
                           << ") with max-iters = " << max_iter
                           << " and tol = " << tolerance
                           << (doPreconditioning ? " with precond " : " without precond ")
                           << ", nrow = " << nrow 
                           << " on ( " << A.geom->npx << " x " << A.geom->npy << " x " << A.geom->npz
                           << " ) MPI grid "
                           << std::endl;
  }
  *flops = 0.0;
  double flops_gmg  = 0.0;
  double flops_spmv = 0.0;
  double flops_orth = 0.0;
  global_int_t numSpMVs_MG = 1+(A.mgData->numberOfPresmootherSteps + A.mgData->numberOfPostsmootherSteps);
  niters = 0;
  bool converged = false;
  while (niters <= max_iter && !converged) {
    // p is of length ncols, copy x to p for sparse MV operation
    CopyVector(x, p);
    TICK(); ComputeSPMV(A, p, Ap); TOCK(t3); flops_spmv += (2*A.totalNumberOfNonzeros); // Ap = A*p
    TICK(); ComputeWAXPBY(nrow, one, b, -one, Ap, r, A.isWaxpbyOptimized); TOCK(t2); *flops += (2*Nrow); // r = b - Ax (x stored in p)
    TICK(); ComputeDotProduct(nrow, r, r, normr, t4, A.isDotProductOptimized); *flops += (2*Nrow); TOCK(t1);
    normr = sqrt(normr);
    GetVector(Q, 0, Qj);
    CopyVector(r, Qj);
    //TICK(); ComputeWAXPBY(nrow, zero, Qj, one/normr, Qj, Qj, A.isWaxpbyOptimized); TOCK(t2);
    TICK(); ScaleVectorValue(Qj, one/normr); TOCK(t2); *flops += Nrow;

    // Record initial residual for convergence testing
    if (niters == 0) normr0 = normr;
    if (verbose && A.geom->rank==0) {
      HPCG_fout << "GMRES Residual at the start of restart cycle = "<< normr
                << ", " << normr/normr0 << std::endl;
    }
    if (normr/normr0 <= tolerance) {
      converged = true;
      if (verbose && A.geom->rank==0) HPCG_fout << " > GMRES converged " << std::endl;
    }

    // do forward GS instead of symmetric GS
    bool symmetric = false;

    // Start restart cycle
    int k = 1;
    SetMatrixValue(t, 0, 0, normr);
    while (k <= restart_length && normr/normr0 > tolerance) {
      GetVector(Q, k-1, Qkm1);
      GetVector(Q, k,   Qk);

      TICK();
      if (doPreconditioning) {
        ComputeMG(A, Qkm1, z, symmetric); flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros); // Apply preconditioner
      } else {
        CopyVector(Qkm1, z);              // copy r to z (no preconditioning)
      }
      TOCK(t5); // Preconditioner apply time

      // Qk = A*z
      TICK(); ComputeSPMV(A, z, Qk); flops_spmv += (2*A.totalNumberOfNonzeros); TOCK(t3);


      // orthogonalize z against Q(:,0:k-1), using dots
      bool use_mgs = false;
      TICK();
      if (use_mgs) {
        // MGS2
        for (int j = 0; j < k; j++) {
          // get j-th column of Q
          GetVector(Q, j, Qj);

          alpha = zero;
          for (int i = 0; i < 2; i++) {
            // beta = Qk'*Qj
            TICK(); ComputeDotProduct(nrow, Qk, Qj, beta, t4, A.isDotProductOptimized); TOCK(t1);

            // Qk = Qk - beta * Qj
            TICK(); ComputeWAXPBY(nrow, one, Qk, -beta, Qj, Qk, A.isWaxpbyOptimized); TOCK(t2);
            alpha += beta;
          }
          SetMatrixValue(H, j, k-1, alpha);
        }
      } else {
        // CGS2
        GetMultiVector(Q, 0, k-1, P);
        ComputeGEMVT (nrow, k,  one, P, Qk, zero, h, A.isGemvOptimized); // h = Q(1:k)'*q(k+1)
        ComputeGEMV  (nrow, k, -one, P, h,  one, Qk, A.isGemvOptimized); // h = Q(1:k)'*q(k+1)
        for(int i = 0; i < k; i++) {
          SetMatrixValue(H, i, k-1, h.values[i]);
        }
        // reorthogonalize
        ComputeGEMVT (nrow, k,  one, P, Qk, zero, h, A.isGemvOptimized); // h = Q(1:k)'*q(k+1)
        ComputeGEMV  (nrow, k, -one, P, h,  one, Qk, A.isGemvOptimized); // h = Q(1:k)'*q(k+1)
        for(int i = 0; i < k; i++) {
          AddMatrixValue(H, i, k-1, h.values[i]);
        }
      }
      TOCK(t6); // Ortho time
      flops_orth += (2*k*Nrow);
      // beta = norm(Qk)
      TICK(); ComputeDotProduct(nrow, Qk, Qk, beta, t4, A.isDotProductOptimized); *flops += (2*Nrow); TOCK(t1);
      beta = sqrt(beta);

      // Qk = Qk / beta
      //TICK(); ComputeWAXPBY(nrow, zero, Qk, one/beta, Qk, Qk, A.isWaxpbyOptimized); TOCK(t2);
      TICK(); ScaleVectorValue(Qk, one/beta); *flops += Nrow; TOCK(t2);
      SetMatrixValue(H, k, k-1, beta);

      // Given's rotation
      for(int j = 0; j < k-1; j++){
        double cj = GetMatrixValue(cs, j, 0);
        double sj = GetMatrixValue(ss, j, 0);
        double h1 = GetMatrixValue(H, j,   k-1);
        double h2 = GetMatrixValue(H, j+1, k-1);

        SetMatrixValue(H, j+1, k-1, -sj * h1 + cj * h2);
        SetMatrixValue(H, j,   k-1,  cj * h1 + sj * h2);
      }

      double f = GetMatrixValue(H, k-1, k-1);
      double g = GetMatrixValue(H, k,   k-1);

      double f2 = f*f;
      double g2 = g*g;
      double fg2 = f2 + g2;
      double D1 = one / sqrt(f2*fg2);
      double cj = f2*D1;
      fg2 = fg2 * D1;
      double sj = f*D1*g;
      SetMatrixValue(H, k-1, k-1, f*fg2);
      SetMatrixValue(H, k,   k-1, zero);

      double v1 = GetMatrixValue(t, k-1, 0);
      double v2 = -v1*sj;
      SetMatrixValue(t, k,   0, v2);
      SetMatrixValue(t, k-1, 0, v1*cj);

      SetMatrixValue(ss, k-1, 0, sj);
      SetMatrixValue(cs, k-1, 0, cj);

      normr = std::abs(v2);
      if (verbose && A.geom->rank==0 && (k%print_freq == 0 || k+1 == restart_length)) {
        HPCG_fout << "GMRES Iteration = "<< k << " (" << niters << ")   Scaled Residual = "
                  << normr << " / " << normr0 << " = " << normr/normr0 << std::endl;
      }
      niters ++;
      k ++;
    } // end of restart-cycle
    // prepare to restart
    if (verbose && A.geom->rank==0) {
      HPCG_fout << "GMRES restart: k = "<< k << " (" << niters << ")" << std::endl;
    }
    // > update x
    ComputeTRSM(k-1, one, H, t);
    if (doPreconditioning) {
      ComputeGEMV(nrow, k-1, one, Q, t, zero, r, A.isGemvOptimized); *flops += (2*Nrow*(k-1)); // r = Q*t
      TICK();
      ComputeMG(A, r, z, symmetric); flops_gmg += (2*numSpMVs_MG*A.totalNumberOfMGNonzeros);      // z = M*r
      TOCK(t5); // Preconditioner apply time
      TICK(); ComputeWAXPBY(nrow, one, x, one, z, x, A.isWaxpbyOptimized); TOCK(t2); *flops += (2*Nrow); // x += z
    } else {
      ComputeGEMV (nrow, k-1, one, Q, t, one, x, A.isGemvOptimized); *flops += (2*Nrow*(k-1)); // x += Q*t
    }
  } // end of outer-loop


  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
//#ifndef HPCG_NO_MPI
//  times[6] += t6; // exchange halo time
//#endif
  times[0] += mytimer() - t_begin;  // Total time. All done...
  if (verbose && A.geom->rank==0) {
    HPCG_fout << " > nnz(A)  : " << A.totalNumberOfNonzeros << std::endl;
    HPCG_fout << " > nnz(MG) : " << A.totalNumberOfMGNonzeros << " (" << numSpMVs_MG << ")" << std::endl;
    HPCG_fout << " > SpMV : " << (flops_spmv / 1000000000.0) << " / " << t3 << " = "
                              << (flops_spmv / 1000000000.0) / t3 << " Gflop/s" << std::endl;
    HPCG_fout << " > GMG  : " << (flops_gmg  / 1000000000.0) << " / " << t5 << " = "
                              << (flops_gmg  / 1000000000.0) / t5 << " Gflop/s" << std::endl;
    HPCG_fout << " > Orth : " << (flops_orth / 1000000000.0) << " / " << t6 << " = "
                              << (flops_orth / 1000000000.0) / t6 << " Gflop/s" << std::endl;
    HPCG_fout << std::endl;
  }
  *flops += flops_gmg;
  *flops += flops_spmv;
  *flops += flops_orth;

  DeleteDenseMatrix(H);
  DeleteDenseMatrix(h);
  DeleteDenseMatrix(t);
  DeleteDenseMatrix(cs);
  DeleteDenseMatrix(ss);
  DeleteMultiVector(Q);

  return 0;
}


/* --------------- *
 * specializations *
 * --------------- */

template
int GMRES< SparseMatrix<double>, CGData<double>, Vector<double> >
  (SparseMatrix<double> const&, CGData<double>&, Vector<double> const&, Vector<double>&,
   const int, const int, double, int&, double&, double&, double*, double*, bool);

template
int GMRES< SparseMatrix<float>, CGData<float>, Vector<float> >
  (SparseMatrix<float> const&, CGData<float>&, Vector<float> const&, Vector<float>&,
   const int, const int, float, int&, float&, float&, double*, double*, bool);
