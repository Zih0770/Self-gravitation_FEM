cmake_minimum_required(VERSION 3.15)

project(Self_gravitation_FEM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# No in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "Use out-of-source build: cmake -S . -B build")
endif()

# Put executables in ./bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# ---- User-provided prefixes ----
set(MFEM_DIR "" CACHE PATH "Path containing MFEMConfig.cmake")
set(mfemElasticity_DIR "" CACHE PATH "Path containing mfemElasticityConfig.cmake")

if(MFEM_DIR)
  list(PREPEND CMAKE_PREFIX_PATH "${MFEM_DIR}")
endif()
if(mfemElasticity_DIR)
  list(PREPEND CMAKE_PREFIX_PATH "${mfemElasticity_DIR}")
endif()

message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

# ---- Dependencies ----
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(MFEM CONFIG REQUIRED)
link_directories(${MFEM_DIR}/fetch/gslib/lib)
find_package(mfemElasticity CONFIG REQUIRED)

# MFEM target name normalization
if(TARGET mfem AND NOT TARGET MFEM::mfem)
  add_library(MFEM::mfem ALIAS mfem)
endif()

# mfemElasticity target normalization
if(TARGET mfemElasticity::mfemElasticity)
  set(MFEM_ELASTICITY_TARGET mfemElasticity::mfemElasticity)
elseif(TARGET mfemElasticity)
  set(MFEM_ELASTICITY_TARGET mfemElasticity)
else()
  message(FATAL_ERROR "mfemElasticity found but no usable target exported")
endif()

# ---- Subdirectories ----
add_subdirectory(meshing)
add_subdirectory(examples)

