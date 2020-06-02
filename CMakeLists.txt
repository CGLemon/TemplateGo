cmake_minimum_required (VERSION 3.9)
project (TemplateGo)

set(USE_BACKEND "CUDA")
if(USE_BACKEND STREQUAL "CUDA")
	message(STATUS "Using CUDA backend")
	enable_language(CUDA)
	aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda CUDA_SRCS)
	add_library(CudaKernel ${CUDA_SRCS})
	#target_compile_features(CudaKernel PUBLIC cxx_std_14)
	set(CUDA_STANDARD 14)

	find_package(CUDA REQUIRED)
	#find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES cuda/include include)
	#if((NOT CUDNN_INCLUDE_DIR))
	#	message(ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is. ${ColorReset}")
	#else()
	#	message("Find cudnn")
	#endif()
	#find_library(CUDNN_LIBRARY libcudnn.so PATHS /usr/local/cuda/lib64 /opt/cuda/lib64)
	#include_directories(${CUDNN_INCLUDE_DIR})
	include_directories(${CUDA_INCLUDE_DIRS})

else()
	message(STATUS "Using CPU backend")

endif()

set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(CMAKE_CXX_FLAGS "-ffast-math -O3  ${CMAKE_CXX_FLAGS}")


find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)


if (BLAS_BACKEND STREQUAL "EIGEN")
	include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen)
	add_definitions(-DUSE_EIGEN)
	add_definitions(-DUSE_BLAS)
	message(STATUS "Use Eigen backend")
elseif(BLAS_BACKEND STREQUAL "OPENBLAS") 
	message(STATUS "Use OpenBlas backend")
	message("Looking for system BLAS/OpenBLAS library.")
	find_package(BLAS REQUIRED)
	find_path(BLAS_INCLUDE_DIRS openblas_config.h
	/usr/include
	/usr/local/include
	/usr/include/openblas
	/opt/OpenBLAS/include
	/usr/include/x86_64-linux-gnu
	$ENV{BLAS_HOME}/include)
	add_definitions(-DUSE_OPENBLAS)
	add_definitions(-DUSE_BLAS)
	if((UNIX AND NOT APPLE) OR WIN32)
		include_directories(${BLAS_INCLUDE_DIRS})
	endif()
	if(APPLE)
		include_directories("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers")
	endif()
else()
	message(STATUS "Using built-in matrix backend.")
endif()

include_directories(${IncludePath})
include_directories(${ZLIB_INCLUDE_DIRS})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/blas CPUBLAS_SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src DIR_SRCS)
add_executable(TemplateGo ${DIR_SRCS} ${CPUBLAS_SRCS})

target_link_libraries(TemplateGo ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(TemplateGo ${ZLIB_LIBRARIES})
target_link_libraries(TemplateGo ${BLAS_LIBRARIES})
target_link_libraries(TemplateGo CudaKernel)
if(USE_BACKEND STREQUAL "CUDA") 
	target_link_libraries(TemplateGo ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
endif()
