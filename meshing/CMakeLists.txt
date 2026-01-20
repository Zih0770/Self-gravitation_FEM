find_package(gmsh CONFIG REQUIRED)

add_library(meshing common.cpp)
target_link_libraries(meshing PUBLIC gmsh::shared)

add_executable(benchmark benchmark.cpp )
target_link_libraries(benchmark meshing)
