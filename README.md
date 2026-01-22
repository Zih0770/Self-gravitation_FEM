This repository serves as the code basis supporting the results in the paper...on self-gravitation computations in an infinite domain. Two approaches are implemented to treat the unbounded domain: the Dirichlet-to-Neumann (DtN) mapping and the multipole expansion method. The C++ classes for implementing these two approaches were developed in the library **mfemElasticity**.  

Three examples are provided for computing within different configurations: 
- examples/ex1: offset sphere benchmark  
- examples/ex2: PREM Earth model 
- examples/ex3: Phobos model 

## Compilation and Dependencies

To compile the example codes, the finite-element library **MFEM** (v4.7 or later) and its extension library **mfemElasticity** are required. Both libraries should be built with MPI enabled. MFEM should be built with **gslib** enabled, as gslib is used for solution interpolation in `ex2` and `ex3`. To compile the meshing codes, the library **gmsh** is required.

After the dependencies are built, the codes in this repository can be compiled using CMake as follows:

```bash
mkdir build
cmake -S . -B build \
  -DMFEM_DIR=/path/to/mfem/build-or-install \
  -DmfemElasticity_DIR=/path/to/mfemElasticity/build-or-install
cmake --build build -j
