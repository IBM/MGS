cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project(ParBrain)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O2 -fopenmp -DOMP")
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O2")
set(CMAKE_LD_FLAGS ${CMAKE_LD_FLAGS} "-lm")


# Add local CMake directory to CMake modules path to load project-specific modules. 
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(ConvertBinToVtu ./src/bin_to_vtu.cxx)
target_link_libraries(ConvertBinToVtu 
	${VTK_LIBRARIES})
