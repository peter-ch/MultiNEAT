cmake_minimum_required(VERSION 2.8.10)

project(MultiNEAT)

option(PYTHON_BINDINGS "Python bindings" OFF)
option(NODE_BINDINGS "Node bindings" OFF)

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS random serialization REQUIRED)

SET(MultiNEAT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB MultiNEAT_SRCS ${MultiNEAT_SOURCE_DIR}/*.cpp)
file(GLOB MultiNEAT_HDRS ${MultiNEAT_SOURCE_DIR}/*.h)

include_directories(${Boost_INCLUDE_DIRS})
add_library(MultiNEAT ${MultiNEAT_SRCS} ${MultiNEAT_HDRS})


if (PYTHON_BINDINGS)
	add_subdirectory(bindings/python)
endif()
