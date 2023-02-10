//This file should contain MPI utils only!!

#ifndef HPGMP_UTILS_HPP
#define HPGMP_UTILS_HPP

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "DataTypes.hpp"

// MpiTypeTraits (from Teuchos)
template<class T>
class MpiTypeTraits {
public:
  static MPI_Datatype getType () {
    return MPI_DATATYPE_NULL;
  }
};

//! Specialization for T = double (from Teuchos)
template<>
class MpiTypeTraits<double> {
public:
  //! MPI_Datatype corresponding to the type T.
  static MPI_Datatype getType () {
    return MPI_DOUBLE;
  }
  static MPI_Op getSumOp () {
    return MPI_SUM;
  }
};

//! Specialization for T = float (from Teuchos).
template<>
class MpiTypeTraits<float> {
public:
  //! MPI_Datatype corresponding to the type T.
  static MPI_Datatype getType () {
    return MPI_FLOAT;
  }
  static MPI_Op getSumOp () {
    return MPI_SUM;
  }
};

#if defined(HPCG_WITH_KOKKOSKERNELS) & defined(KOKKOS_HALF_IS_FULL_TYPE_ON_ARCH) // if arch does not support half, then half = float
//! Specialization for T = half
template<>
class MpiTypeTraits<half_t> {
public:
  //! MPI_Datatype corresponding to the type T.
  static MPI_Datatype getType () {
    return HPGMP_MPI_HALF;
  }
  static MPI_Op getSumOp () {
    return MPI_SUM_HALF;
  }
};
#endif

#endif // ifndef HPCG_NO_MPI
#endif
