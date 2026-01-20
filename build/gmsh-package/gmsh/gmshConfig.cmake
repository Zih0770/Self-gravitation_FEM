if(NOT TARGET gmsh::shared)
  add_library(gmsh::shared UNKNOWN IMPORTED)
  set_target_properties(gmsh::shared PROPERTIES
    IMPORTED_LOCATION "/home/sssou/local/gmsh/lib/libgmsh.so"
    INTERFACE_INCLUDE_DIRECTORIES "/home/sssou/local/gmsh/include"
  )
endif()
