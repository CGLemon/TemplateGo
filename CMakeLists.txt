cmake_minimum_required (VERSION 3.9)
project (TemplateGo)

if (USE_OPENMP) 
	set(CMAKE_CXX_FLAGS "-fopenmp  ${CMAKE_CXX_FLAGS}")
endif()


if(GPU_BACKEND STREQUAL "CUDA")
	message(STATUS "Using CUDA backend")
	add_definitions(-DUSE_CUDA)

	enable_language(CUDA)
	aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda CUDA_SRCS)

	set(CUDA_STANDARD 14)

else()
	message(STATUS "Using CPU backend")

endif()

set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(CMAKE_CXX_FLAGS "-ffast-math -O3  ${CMAKE_CXX_FLAGS}")

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


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/blas CPUBLAS_SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src DIR_SRCS)
add_executable(TemplateGo ${DIR_SRCS} ${CPUBLAS_SRCS} ${CUDA_SRCS})

target_link_libraries(TemplateGo ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(TemplateGo ${BLAS_LIBRARIES})
if(GPU_BACKEND STREQUAL "CUDA")
	target_compile_definitions(TemplateGo PRIVATE USE_CUDA_BACKEND)
	find_package(CUDA REQUIRED)
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
	target_link_libraries(TemplateGo ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
endif()

if(USE_ZLIB)
	find_package(ZLIB REQUIRED)
	include_directories(${ZLIB_INCLUDE_DIRS})
	target_link_libraries(TemplateGo ${ZLIB_LIBRARIES})
	add_definitions(-DUSE_ZLIB)
	message(STATUS "Using zlib.")
	message("The program can read gzip file")
else()
	message(STATUS "whiout using zlib.")
	message("Because whiout using zlib.")
	message("The program can't read gzip file.")
endif()


