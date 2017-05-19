cmake_minimum_required(VERSION 3.3)
project(MultiNEAT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(PythonLibs REQUIRED)
include(FindPythonLibs)

set(Boost_USE_STATIC_LIBS       OFF) # only find static libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost COMPONENTS date_time system filesystem python3 serialization)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

#include_directories(${PYTHON_INCLUDE_DIRS})

#add_definitions(-DUSE_BOOST_PYTHON)

set(SOURCE_FILES
    src/Assert.h
    src/Genes.h
    src/Genome.cpp
    src/Genome.h
    src/Innovation.cpp
    src/Innovation.h
    src/Main.cpp
    src/NeuralNetwork.cpp
    src/NeuralNetwork.h
    src/Parameters.cpp
    src/Parameters.h
    src/PhenotypeBehavior.cpp
    src/PhenotypeBehavior.h
    src/Population.cpp
    src/Population.h
    src/PythonBindings.cpp
    src/PythonBindings.h
    src/Random.cpp
    src/Random.h
    src/Species.cpp
    src/Species.h
    src/Substrate.cpp
    src/Substrate.h
    src/Utils.cpp
    src/Utils.h src/Traits.h src/Traits.cpp)

add_executable(MultiNEAT ${SOURCE_FILES})
target_link_libraries(MultiNEAT ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} pthread)