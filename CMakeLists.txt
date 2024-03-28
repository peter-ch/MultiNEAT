cmake_minimum_required(VERSION 3.3)
project(MultiNEAT)


if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")

find_package(PythonLibs REQUIRED)
include(FindPythonLibs)
#set(Boost_Path "C:/boost/include")

#set(BOOST_ROOT "C:\\Boost")
#set(BOOSTROOT "C:\\Boost")

add_definitions("-DMS_WIN64")

set(Boost_USE_STATIC_LIBS       OFF) # only find static libs
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME   OFF)

find_package(BOOST COMPONENTS REQUIRED
        system
        python
        numpy
        date_time
        filesystem
        serialization)

if(Boost_FOUND)
    message(STATUS "It works!")
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories("C:/Boost/lib")
endif()


include_directories(${PYTHON_INCLUDE_DIRS})

add_definitions(-DUSE_BOOST_PYTHON)
#add_definitions(-DUSE_BOOST_NUMPY)
add_definitions(-DUSE_BOOST_RANDOM)
#add_definitions(-DVDEBUG)


add_executable(MultiNEAT ${PROJECT_SOURCE_DIR}/src/Assert.h
                         ${PROJECT_SOURCE_DIR}/src/Genes.h
                         ${PROJECT_SOURCE_DIR}/src/Genome.cpp
                         ${PROJECT_SOURCE_DIR}/src/Genome.h
                         ${PROJECT_SOURCE_DIR}/src/Innovation.cpp
                         ${PROJECT_SOURCE_DIR}/src/Innovation.h
                         ${PROJECT_SOURCE_DIR}/src/Main.cpp
                         ${PROJECT_SOURCE_DIR}/src/NeuralNetwork.cpp
                         ${PROJECT_SOURCE_DIR}/src/NeuralNetwork.h
                         ${PROJECT_SOURCE_DIR}/src/Parameters.cpp
                         ${PROJECT_SOURCE_DIR}/src/Parameters.h
                         ${PROJECT_SOURCE_DIR}/src/PhenotypeBehavior.cpp
                         ${PROJECT_SOURCE_DIR}/src/PhenotypeBehavior.h
                         ${PROJECT_SOURCE_DIR}/src/Population.cpp
                         ${PROJECT_SOURCE_DIR}/src/Population.h
                         ${PROJECT_SOURCE_DIR}/src/PythonBindings.cpp
                         ${PROJECT_SOURCE_DIR}/src/Random.cpp
                         ${PROJECT_SOURCE_DIR}/src/Random.h
                         ${PROJECT_SOURCE_DIR}/src/Species.cpp
                         ${PROJECT_SOURCE_DIR}/src/Species.h
                         ${PROJECT_SOURCE_DIR}/src/Substrate.cpp
                         ${PROJECT_SOURCE_DIR}/src/Substrate.h
                         ${PROJECT_SOURCE_DIR}/src/Utils.cpp
                         ${PROJECT_SOURCE_DIR}/src/Utils.h
                         ${PROJECT_SOURCE_DIR}/src/Traits.h
                         ${PROJECT_SOURCE_DIR}/src/Traits.cpp)

target_link_libraries(MultiNEAT
        "ws2_32"
        ${BOOST_LIBRARIES}
        ${PYTHON_LIBRARIES})
