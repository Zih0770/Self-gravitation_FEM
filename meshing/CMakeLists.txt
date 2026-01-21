find_package(gmsh CONFIG REQUIRED)

add_library(meshing common.cpp)
target_link_libraries(meshing PUBLIC gmsh::shared)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark meshing)

add_executable(prem prem.cpp)
target_link_libraries(prem meshing)

add_executable(Phobos Phobos.cpp)
target_link_libraries(Phobos meshing)
