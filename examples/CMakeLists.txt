add_executable(ex1 ex1.cpp)
target_link_libraries(ex1
  PRIVATE
    MPI::MPI_CXX
    MFEM::mfem
    ${MFEM_ELASTICITY_TARGET}
)

add_executable(ex2 ex2.cpp)
target_link_libraries(ex2
  PRIVATE
    MPI::MPI_CXX
    MFEM::mfem
    ${MFEM_ELASTICITY_TARGET}
)

add_executable(ex3 ex3.cpp)
target_link_libraries(ex3
  PRIVATE
    MPI::MPI_CXX
    MFEM::mfem
    ${MFEM_ELASTICITY_TARGET}
)

