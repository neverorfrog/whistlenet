cmake_minimum_required(VERSION 3.19)
project(cpp-demo)

# Target with properties that we want to maintain throughout project
add_library(cxx_setup INTERFACE)
# target_compile_options(cxx_setup INTERFACE -Wall -Wpedantic -Wextra)
target_compile_features(cxx_setup INTERFACE cxx_std_17)

set(PYTHON_EXECUTABLE python3)
set(TORCH_PATH $ENV{TORCH_PATH})
set(EXECUTORCH_ROOT ${TORCH_PATH}/executorch)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
set(CMAKE_PREFIX_PATH ${TORCH_PATH}/libtorch)

###########################################################################
# Potentially this is the only line you need to change from this CMakeLists
set(ops cnn_ops)
###########################################################################

add_subdirectory(${ops})

# torch library
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# executorch library
add_library(executorch STATIC IMPORTED GLOBAL)
set_property(TARGET executorch PROPERTY IMPORTED_LOCATION "${EXECUTORCH_ROOT}/cmake-out/libexecutorch.a")
target_include_directories(executorch INTERFACE "${TORCH_PATH}")

# operations yaml generation
add_custom_target(kernel_oplist ALL
COMMENT 
  "Generating selected_operators.yaml for ops"
COMMAND 
  ${PYTHON_EXECUTABLE} -m codegen.tools.gen_oplist 
  --output_path=${CMAKE_CURRENT_BINARY_DIR}/${ops}/selected_operators.yaml
  --ops_schema_yaml_path=${CMAKE_CURRENT_LIST_DIR}/${ops}/ops.yaml
WORKING_DIRECTORY 
  ${EXECUTORCH_ROOT})

# kernel bindings generation
add_custom_target(kernel_bindings ALL
COMMENT 
    "Generating code for kernel registration"
COMMAND
    ${PYTHON_EXECUTABLE} -m torchgen.gen_executorch
    --source-path=${EXECUTORCH_ROOT}/codegen
    --install-dir=${CMAKE_CURRENT_BINARY_DIR}/${ops}
    --tags-path=${TORCH_ROOT}/aten/src/ATen/native/tags.yaml
    --aten-yaml-path=${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml
    --op-selection-yaml-path=${CMAKE_CURRENT_BINARY_DIR}/${ops}/selected_operators.yaml
    --functions-yaml-path=${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
DEPENDS
  kernel_oplist
BYPRODUCTS 
    ${CMAKE_CURRENT_BINARY_DIR}/${ops}/Functions.h
    ${CMAKE_CURRENT_BINARY_DIR}/${ops}/RegisterCodegenUnboxedKernelsEverything.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/${ops}/NativeFunctions.h
WORKING_DIRECTORY 
    ${EXECUTORCH_ROOT})

# operations library (from yaml and bindings)
add_library(${ops})
add_dependencies(${ops} kernel_bindings)
target_sources(${ops} PUBLIC #public in order to be visible to executorch for registering
  ${CMAKE_CURRENT_BINARY_DIR}/${ops}/RegisterCodegenUnboxedKernelsEverything.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/${ops}/Functions.h
  ${CMAKE_CURRENT_BINARY_DIR}/${ops}/NativeFunctions.h)
target_link_libraries(${ops} PRIVATE executorch)
target_link_libraries(${ops} PUBLIC kernel)
target_include_directories(${ops} PUBLIC "${TORCH_PATH}")

# executable
add_executable(example
  ${CMAKE_CURRENT_SOURCE_DIR}/example.cpp
  ${EXECUTORCH_ROOT}/extension/data_loader/file_data_loader.cpp
  ${EXECUTORCH_ROOT}/extension/evalue_util/print_evalue.cpp
)
target_include_directories(example PRIVATE "${TORCH_PATH}")
target_link_libraries(example cxx_setup "${TORCH_LIBRARIES}" executorch ${ops})
target_compile_options(example PUBLIC -Wno-deprecated-declarations -fPIC)