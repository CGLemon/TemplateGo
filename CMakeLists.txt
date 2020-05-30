cmake_minimum_required (VERSION 3.9)
project (TemplateGo)

#set(USE_BACKEND "CUDA")
if(USE_BACKEND STREQUAL "CUDA")
	message("Using CUDA backend")
	enable_language(CUDA)
	aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda CUDA_SRCS)
	add_library(CudaKernel ${CUDA_SRCS})
	#target_compile_features(CudaKernel PUBLIC cxx_std_14)
	set(CUDA_STANDARD 14)

	find_package(CUDA REQUIRED)
	find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES cuda/include include)
	if((NOT CUDNN_INCLUDE_DIR))
		message(ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is. ${ColorReset}")
	else()
		message("Find cudnn")
	endif()
	find_library(CUDNN_LIBRARY libcudnn.so PATHS /usr/local/cuda/lib64 /opt/cuda/lib64)
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})

else()
	message("Using CPU backend")

endif()


set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen")
set(CMAKE_CXX_FLAGS "-ffast-math -O3  ${CMAKE_CXX_FLAGS}")


find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
  
include_directories(${IncludePath})
include_directories(${ZLIB_INCLUDE_DIRS})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/blas CPUBLAS_SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src DIR_SRCS)
add_executable(TemplateGo ${DIR_SRCS} ${CPUBLAS_SRCS})

target_link_libraries(TemplateGo ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(TemplateGo ${ZLIB_LIBRARIES})
target_link_libraries(TemplateGo CudaKernel)
if(USE_BACKEND STREQUAL "CUDA") 
	target_link_libraries(TemplateGo ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
endif()
