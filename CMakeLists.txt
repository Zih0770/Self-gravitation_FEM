cmake_minimum_required(VERSION 3.15)

project(paper_self_gravitation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prohibit in-source builds
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source builds are prohibited. Use: cmake -S . -B build")
endif()

# Put all executables in <repo>/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${cfg}" cfg_uc)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_uc} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endforeach()

option(BUILD_EXAMPLES "Build FEM examples" ON)
option(BUILD_MESHING  "Build meshing tools (gmsh)" ON)

# -----------------------------
# MFEM + mfemElasticity + MPI
# -----------------------------
set(MFEM_DIR "" CACHE PATH "Path to MFEM build/install prefix (or dir containing MFEMConfig.cmake)")
set(mfemElasticity_DIR "" CACHE PATH "Path to mfemElasticity build/install dir (contains mfemElasticityConfig.cmake)")

# Help CONFIG packages be found
foreach(_p MFEM_DIR mfemElasticity_DIR)
  if(DEFINED ${_p} AND NOT "${${_p}}" STREQUAL "")
    list(PREPEND CMAKE_PREFIX_PATH "${${_p}}")
  endif()
endforeach()
message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

# All your codes use MPI
find_package(MPI REQUIRED COMPONENTS CXX)

# MFEM (CONFIG). MFEM might export MFEM::mfem or mfem
message(STATUS "Looking for MFEM ...")
find_package(MFEM CONFIG REQUIRED)
message(STATUS "Found MFEM version: ${MFEM_VERSION}")

# mfemElasticity (CONFIG)
message(STATUS "Looking for mfemElasticity ...")
find_package(mfemElasticity CONFIG REQUIRED)

# Your examples/ wants to link to target name exactly "mfemElasticity".
# If the package exports mfemElasticity::mfemElasticity instead, create an alias.
if(TARGET mfemElasticity::mfemElasticity AND NOT TARGET mfemElasticity)
  add_library(mfemElasticity ALIAS mfemElasticity::mfemElasticity)
endif()

if(NOT TARGET mfemElasticity)
  message(FATAL_ERROR
    "mfemElasticity package was found, but no target named 'mfemElasticity' exists.\n"
    "Either export that target from mfemElasticity, or link examples to mfemElasticity::mfemElasticity."
  )
endif()

# -----------------------------
# Gmsh: provide a gmsh CONFIG package for meshing/
# so meshing/CMakeLists can do: find_package(gmsh CONFIG REQUIRED)
# -----------------------------
if(BUILD_MESHING)
  set(GMSH_INCLUDE_DIR "" CACHE PATH "Path to gmsh include dir containing gmsh.h")
  set(GMSH_LIBRARY "" CACHE FILEPATH "Full path to libgmsh.so (or .a)")

  if(NOT (GMSH_INCLUDE_DIR AND EXISTS "${GMSH_INCLUDE_DIR}/gmsh.h" AND
          GMSH_LIBRARY AND EXISTS "${GMSH_LIBRARY}"))
    message(FATAL_ERROR
      "BUILD_MESHING=ON but gmsh was not found.\n"
      "Set:\n"
      "  -DGMSH_INCLUDE_DIR=/path/to/gmsh/include (contains gmsh.h)\n"
      "  -DGMSH_LIBRARY=/path/to/gmsh/lib/libgmsh.so (or .a)"
    )
  endif()

  message(STATUS "Gmsh includes: ${GMSH_INCLUDE_DIR}")
  message(STATUS "Gmsh library : ${GMSH_LIBRARY}")

  # Create imported target gmsh::shared
  add_library(gmsh::shared UNKNOWN IMPORTED)
  set_target_properties(gmsh::shared PROPERTIES
    IMPORTED_LOCATION "${GMSH_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GMSH_INCLUDE_DIR}"
  )

  # Write a minimal gmshConfig.cmake into the build tree so find_package(gmsh CONFIG) works.
  set(_gmsh_pkg_root "${CMAKE_BINARY_DIR}/gmsh-package")
  set(_gmsh_pkg_dir  "${_gmsh_pkg_root}/gmsh")
  file(MAKE_DIRECTORY "${_gmsh_pkg_dir}")

  file(WRITE "${_gmsh_pkg_dir}/gmshConfig.cmake"
"if(NOT TARGET gmsh::shared)
  add_library(gmsh::shared UNKNOWN IMPORTED)
  set_target_properties(gmsh::shared PROPERTIES
    IMPORTED_LOCATION \"${GMSH_LIBRARY}\"
    INTERFACE_INCLUDE_DIRECTORIES \"${GMSH_INCLUDE_DIR}\"
  )
endif()
")

  # Make sure subdirectories can find it
  list(PREPEND CMAKE_PREFIX_PATH "${_gmsh_pkg_root}")
endif()

# -----------------------------
# Subdirectories
# -----------------------------
if(BUILD_MESHING)
  add_subdirectory(meshing)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

