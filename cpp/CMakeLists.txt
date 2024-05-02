cmake_minimum_required(VERSION 3.16)

#Check for build types
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

#Compile options
add_library(cxx_setup INTERFACE)
target_compile_options(cxx_setup INTERFACE -Wall -Wpedantic -Wextra)
target_compile_features(cxx_setup INTERFACE cxx_std_17)
if(CMAKE_CXX_COMPILER_LOADED)
    message(STATUS "Path: ${CMAKE_CXX_COMPILER}")
    message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/executables")

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Blue        "${Esc}[34m")
endif()

project(cpptest VERSION 0.1 DESCRIPTION "Sound registration and elaboration through portaudio and kissfft" LANGUAGES CXX) 

include(CMakePrintHelpers)
cmake_print_variables(PROJECT_SOURCE_DIR)
cmake_print_variables(PROJECT_BINARY_DIR)
set(INCLUDE_DIRECTORY ${PROJECT_SOURCE_DIR}/third-party)

# Find PortAudio
list(APPEND CMAKE_PREFIX_PATH external/portaudio)
find_package(PortAudio CONFIG REQUIRED)
include_directories(external/portaudio/include)
link_directories(external/portaudio/lib)

# Find KissFFT
# list(APPEND CMAKE_PREFIX_PATH ${INCLUDE_DIRECTORY}/kissfft)
# find_package(kissfft CONFIG REQUIRED)
# include_directories(${INCLUDE_DIRECTORY}/kissfft/include)
# link_directories(${INCLUDE_DIRECTORY}/kissfft/lib)

# Source files
add_library(recording STATIC AudioData.h AudioRecorder.cpp AudioPlayer.cpp)
target_link_libraries(recording PUBLIC portaudio cxx_setup asound)
add_executable(test ${PROJECT_SOURCE_DIR}/test.cpp)
target_link_libraries(test cxx_setup portaudio recording asound)