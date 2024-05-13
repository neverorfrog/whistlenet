set(kernel_sources
      ${ET_ROOT}/kernels/portable/cpu/op_addmm.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_alias_copy.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_permute_copy.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_leaky_relu.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_to_copy.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_convolution.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_max_pool2d_with_indices.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_view_copy.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_softmax.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_argmax.cpp
      ${ET_ROOT}/kernels/portable/cpu/op_squeeze_copy.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/broadcast_util.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/activation_ops_util.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/advanced_index_util.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/copy_ops_util.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/index_util.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/kernel_ops_util.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/matmul_ops_util.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/normalization_ops_util.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/reduce_util.cpp
      ${ET_ROOT}/kernels/portable/cpu/util/repeat_util.cpp
)