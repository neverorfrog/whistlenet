# ------------------ Python files IMPORTS -----------------------
include(${ET_ROOT}/build/Codegen.cmake)
execute_process(
    COMMAND
        "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib;print(get_python_lib())"
    OUTPUT_VARIABLE site-packages-out
    ERROR_VARIABLE site-packages-out-error
    RESULT_VARIABLE site-packages-result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(GLOB_RECURSE _codegen_tools_srcs "${ET_ROOT}/codegen/tools/*.py")
file(GLOB_RECURSE _codegen_templates "${ET_ROOT}/codegen/templates/*")
file(GLOB_RECURSE _torchgen_srcs "${site-packages-out}/torchgen/*.py")
# ----------------- END IMPORTS ---------------------------------

# ------- OPTIONS ----------------------------------------------
option(EXECUTORCH_SELECT_OPS_YAML "Register all the ops from a given yaml file" ON)
option(EXECUTORCH_SELECT_OPS_LIST "Register a list of ops, separated by comma" OFF)
option(EXECUTORCH_SELECT_ALL_OPS "Whether to register all ops defined in portable kernel library." OFF)
# ------- END OPTIONS ------------------------------------------

# ------------------ VAR SETTING ------------------------------
set(_kernel_lib)
set(_out_dir ${CMAKE_CURRENT_BINARY_DIR}/${_lib_name})
file(MAKE_DIRECTORY ${_out_dir})
set(_oplist_yaml ${_out_dir}/selected_operators.yaml)
# ------------------ END VAR SETTING ----------------------------


# selected operations yaml generation
add_custom_target(_kernel_oplist ALL
COMMENT 
    "Generating selected_operators.yaml for ${_lib_name}"
COMMAND 
    ${PYTHON_EXECUTABLE} -m codegen.tools.gen_oplist 
    --output_path=${_oplist_yaml}
    --ops_schema_yaml_path=${CMAKE_CURRENT_LIST_DIR}/${_lib_name}/op.yaml
DEPENDS 
    ${_codegen_tools_srcs}
WORKING_DIRECTORY 
    ${ET_ROOT}
)

set(EXECUTORCH_SRCS_FILE "${ET_BUILD_PATH}/executorch_srcs.cmake")
include(${ET_ROOT}/build/Utils.cmake)
extract_sources(${EXECUTORCH_SRCS_FILE})
include(${EXECUTORCH_SRCS_FILE})

# kernel library in c++ from selected operations
if(EXECUTORCH_SELECT_OPS_YAML)
  include(${_lib_name}/kernel.cmake)
  add_library(custom_kernel ${kernel_sources} ${EXECUTORCH_SRCS_FILE})
  target_include_directories(custom_kernel PUBLIC ${TORCH_PATH} ${ET_ROOT})
  target_link_libraries(custom_kernel PUBLIC executorch)

  list(APPEND _kernel_lib custom_kernel)
else()
  list(APPEND _kernel_lib portable_kernels)
endif()

# kernel bindings generation
add_custom_target(_kernel_bindings ALL
COMMENT 
    "Generating code for kernel registration"
COMMAND
    ${PYTHON_EXECUTABLE} -m torchgen.gen_executorch
    --source-path=${ET_ROOT}/codegen
    --install-dir=${_out_dir}
    --tags-path=${site-packages-out}/torchgen/packaged/ATen/native/tags.yaml
    --aten-yaml-path=${site-packages-out}/torchgen/packaged/ATen/native/native_functions.yaml
    --op-selection-yaml-path=${_oplist_yaml}
    --functions-yaml-path=${ET_ROOT}/kernels/portable/functions.yaml
DEPENDS
    ${_kernel_oplist} ${_codegen_templates} ${_torchgen_srcs} 
BYPRODUCTS 
    ${_out_dir}/Functions.h
    ${_out_dir}/RegisterCodegenUnboxedKernelsEverything.cpp
    ${_out_dir}/NativeFunctions.h
WORKING_DIRECTORY 
    ${ET_ROOT}
)

# operator library generation
add_library(${_lib_name})
target_sources(${_lib_name} PUBLIC #public in order to be visible to executorch for registering
        ${_out_dir}/RegisterCodegenUnboxedKernelsEverything.cpp
        ${_out_dir}/Functions.h 
        ${_out_dir}/NativeFunctions.h
)
target_link_libraries(${_lib_name} PRIVATE executorch)
target_link_libraries(${_lib_name} PUBLIC ${_kernel_lib})
target_include_directories(${_lib_name} PUBLIC "${TORCH_PATH}")

