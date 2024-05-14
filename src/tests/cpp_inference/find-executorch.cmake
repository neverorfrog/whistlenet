find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(Torch_FOUND)
  message(STATUS "${Green}LIB_Torch found${ColourReset}")
endif()

# IMPORT PATHS
list(APPEND CMAKE_PREFIX_PATH
    "${ET_ROOT}/cmake-out"
    "${ET_ROOT}/cmake-out/kernels/portable"
    "${ET_ROOT}/cmake-out/sdk"
    "${ET_ROOT}/cmake-out/extension/data_loader"
)

# REQUIRED LIBS
set(required_lib_list 
    executorch 
    executorch_no_prim_ops 
    portable_kernels
    portable_ops_lib
    etdump
    bundled_program
    extension_data_loader
    XNNPACK
    cpuinfo
    pthreadpool
)
foreach(lib ${required_lib_list})
    set(lib_var "LIB_${lib}")
    find_library(${lib_var} ${lib} REQUIRED CMAKE_FIND_ROOT_PATH_BOTH)
    if ("${${lib_var}}" STREQUAL "${lib_var}-NOTFOUND")
        message("${Red}${lib} library is not found${ColourReset}")
    else()
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${${lib_var}}")
        target_include_directories(${lib} INTERFACE ${_root})
        message(STATUS "${Green}${lib_var} found at location ${${lib_var}}${ColourReset}")
    endif()
endforeach()

target_link_libraries(executorch INTERFACE executorch_no_prim_ops)
target_include_directories(executorch INTERFACE "${TORCH_PATH}")

# OPTIONAL LIBS
# set(lib_list
#     flatccrt
#     mpsdelegate
#     qnn_executorch_backend
#     extension_module
#     xnnpack_backend
#     vulkan_backend
#     optimized_kernels
#     cpublas
#     eigen_blas
#     optimized_ops_lib
#     optimized_native_cpu_ops_lib
#     quantized_kernels
#     quantized_ops_lib
# )

# foreach(lib ${lib_list})
#     # Name of the variable which stores result of the find_library search
#     set(lib_var "LIB_${lib}")
#     find_library(
#       ${lib_var} ${lib}
#       PATHS "${ET_ROOT}"
#       CMAKE_FIND_ROOT_PATH_BOTH)
#       if ("${${lib_var}}" STREQUAL "${lib_var}-NOTFOUND")
#           message("${Red}${lib} library is not found${ColourReset}")
#       else()
#           message(STATUS "${Green}${lib_var} found${ColourReset}")
#           if("${lib}" STREQUAL "extension_module" AND (NOT CMAKE_TOOLCHAIN_IOS))
#               add_library(${lib} SHARED IMPORTED)
#           else()
#               # Building a share library on iOS requires code signing, so it's easier to
#               # keep all libs as static when CMAKE_TOOLCHAIN_IOS is used
#               add_library(${lib} STATIC IMPORTED)
#           endif()
#           set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${${lib_var}}")
#           target_include_directories(${lib} INTERFACE ${_root})
#       endif()
#   endforeach()

