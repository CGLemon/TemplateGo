cmake_minimum_required (VERSION 3.0)
project (Train)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Torch REQUIRED)

add_definitions(-DUSE_TORCH)
aux_source_directory(./src DIR_SRCS)
add_executable(Train ${DIR_SRCS})
target_link_libraries(Train "${TORCH_LIBRARIES}")

##
## mkdir build
## cd build
## cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
## cmake --build . --config Release
##

