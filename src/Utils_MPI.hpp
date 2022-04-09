//This file should contain MPI utils only!!
#ifndef HPCG_NO_MPI

#ifndef HPGMP_UTILS_MPI_HPP
#define HPGMP_UTILS_MPI_HPP

#include <mpi.h>

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
};

//! Specialization for T = float (from Teuchos).
template<>
class MpiTypeTraits<float> {
public:
  //! MPI_Datatype corresponding to the type T.
  static MPI_Datatype getType () {
    return MPI_FLOAT;
  }
};

#endif

#endif // ifndef HPCG_NO_MPI
