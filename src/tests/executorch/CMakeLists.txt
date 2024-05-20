cmake_minimum_required(VERSION 3.19)

# Check for build types
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Compile options
add_library(cxx_setup INTERFACE)
# target_compile_options(cxx_setup INTERFACE -Wall -Wpedantic -Wextra)
target_compile_features(cxx_setup INTERFACE cxx_std_17)
if(CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "Path: ${CMAKE_CXX_COMPILER}")
  message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/executables")

# some extra variables and imports
set(TORCH_PATH $ENV{TORCH_PATH})
set(ET_ROOT ${TORCH_PATH}/executorch)
set(PYTORCH_ROOT ${ET_ROOT}/third-party/pytorch)
set(CMAKE_PREFIX_PATH ${TORCH_PATH}/libtorch)
set(ET_BUILD_PATH ${ET_ROOT}/cmake-out)
set(PYTHON_EXECUTABLE python)

# just for printing
string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Red "${Esc}[31m")
set(Green "${Esc}[32m")
set(Blue "${Esc}[34m")

project(cpp-demo VERSION 0.1 LANGUAGES CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(Torch_FOUND)
  message(STATUS "${Green}LIB_Torch found${ColourReset}")
endif()

list(APPEND CMAKE_PREFIX_PATH
    "${ET_ROOT}/cmake-out"
    "${ET_ROOT}/cmake-out/kernels/portable"
    "${ET_ROOT}/cmake-out/sdk"
    "${ET_ROOT}/cmake-out/extension/data_loader"
)
find_package(executorch REQUIRED)
if(executorch_FOUND)
  message(STATUS "${Green}LIB_Executorch found${ColourReset}")
endif()

link_directories("${ET_ROOT}/cmake-out/extension/runner_util")
find_library(extension_runner_util HINTS "${ET_ROOT}/cmake-out/extension/runner_util")

set(_lib_name "cnn")
include(${CMAKE_CURRENT_LIST_DIR}/kernel.cmake)

# executable
add_executable(main 
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${ET_ROOT}/extension/data_loader/file_data_loader.cpp
  ${ET_ROOT}/extension/evalue_util/print_evalue.cpp
)
target_include_directories(main PRIVATE "${TORCH_PATH}")
target_link_libraries(main cxx_setup "${TORCH_LIBRARIES}" executorch ${_lib_name} extension_runner_util)