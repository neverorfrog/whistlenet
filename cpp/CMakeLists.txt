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

#just for printing
string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Red         "${Esc}[31m")
set(Green       "${Esc}[32m")
set(Blue        "${Esc}[34m")

project(whistlenet VERSION 0.1 LANGUAGES CXX)

# Find PortAudio
list(APPEND CMAKE_PREFIX_PATH external/portaudio)
find_package(PortAudio CONFIG REQUIRED)
include_directories(external/portaudio/include)
link_directories(external/portaudio/lib)
if (PortAudio_FOUND)
    message(STATUS "${Green}PortAudio found${ColourReset}")
else()
    message(FATAL_ERROR "${Red}PortAudio not found${ColourReset}")
endif()

# Find KissFFT
list(APPEND CMAKE_PREFIX_PATH external/kissfft)
find_package(kissfft CONFIG REQUIRED)
include_directories(external/kissfft/include)
link_directories(external/kissfft/lib)
if (kissfft_FOUND)
    message(STATUS "${Green}KissFFT found${ColourReset}")
else()
    message(FATAL_ERROR "${Red}KissFFT not found${ColourReset}")
endif()

add_library(audiolib AudioData.h AudioRecorder.cpp AudioPlayer.cpp)
target_link_libraries(audiolib PUBLIC cxx_setup portaudio asound kissfft-float)
add_executable(test test.cpp)
target_link_libraries(test PUBLIC audiolib)