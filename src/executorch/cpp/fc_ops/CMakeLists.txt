cmake_minimum_required(VERSION 3.19)
project(fc_ops)

set(kernel_utils
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/broadcast_util.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/activation_ops_util.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/advanced_index_util.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/copy_ops_util.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/index_util.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/kernel_ops_util.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/matmul_ops_util.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/normalization_ops_util.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/reduce_util.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/util/repeat_util.cpp)

# kernel library
add_library(kernel)
target_sources(kernel PUBLIC
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_addmm.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_permute_copy.cpp
  ${EXECUTORCH_ROOT}/kernels/portable/cpu/op_relu.cpp
  ${kernel_utils})
target_link_libraries(kernel PRIVATE executorch)
target_include_directories(kernel PUBLIC "${TORCH_PATH}")