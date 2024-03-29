cmake_minimum_required(VERSION 3.6)
project(MultiNEAT)

if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_definitions("-DMS_WIN64")

set(Boost_USE_STATIC_LIBS OFF) # Changed comment to reflect that dynamic libs are actually needed
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

# Make sure you have the correct Boost components, especially boost_python for your Python version (e.g., boost_python38 for Python 3.8)
set(BOOST_ROOT "C:/Boost")
find_package(Boost REQUIRED COMPONENTS 
  system
  python310
  numpy310
  thread
  serialization
  date_time
  filesystem
)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

add_definitions(-DUSE_BOOST_PYTHON)
# Optional: Enable if you use Boost.Numpy in your bindings
add_definitions(-DUSE_BOOST_NUMPY)
add_definitions(-DUSE_BOOST_RANDOM)

# Replace add_executable with add_library, and specify SHARED to create a shared library
add_library(MultiNEAT SHARED
             # List all your source and header files as before
             ${PROJECT_SOURCE_DIR}/src/PythonBindings.cpp # Make sure this file uses BOOST_PYTHON_MODULE
             ${PROJECT_SOURCE_DIR}/src/Assert.h
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
             ${PROJECT_SOURCE_DIR}/src/Random.cpp
             ${PROJECT_SOURCE_DIR}/src/Random.h
             ${PROJECT_SOURCE_DIR}/src/Species.cpp
             ${PROJECT_SOURCE_DIR}/src/Species.h
             ${PROJECT_SOURCE_DIR}/src/Substrate.cpp
             ${PROJECT_SOURCE_DIR}/src/Substrate.h
             ${PROJECT_SOURCE_DIR}/src/Utils.cpp
             ${PROJECT_SOURCE_DIR}/src/Utils.h
             ${PROJECT_SOURCE_DIR}/src/Traits.h
             ${PROJECT_SOURCE_DIR}/src/Traits.cpp
)

# Ensure you don't have a Main.cpp in add_library, as typically Main.cpp contains int main() which is not needed for Python modules

target_link_libraries(MultiNEAT
        "ws2_32"
        ${Boost_LIBRARIES}
        ${Python3_LIBRARIES})

# If your project name and the desired Python module name differ, replace MultiNEAT in set_target_properties with the actual module name
set_target_properties(MultiNEAT PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(MultiNEAT PROPERTIES SUFFIX ".pyd")
else()
    # For Unix-like systems, the default suffix is fine (.so), but ensure the prefix is correct for Python to find it
    set_target_properties(MultiNEAT PROPERTIES PREFIX "lib")
endif()




