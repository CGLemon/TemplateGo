cmake_minimum_required (VERSION 2.8)
project (TemplateGo)



set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen")

add_compile_options(-O3)
add_compile_options(-ffast-math)

find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

include_directories(${IncludePath})
include_directories(${ZLIB_INCLUDE_DIRS})

aux_source_directory(./src DIR_SRCS)
add_executable(TemplateGo ${DIR_SRCS})

target_link_libraries(TemplateGo ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(TemplateGo ${ZLIB_LIBRARIES})
