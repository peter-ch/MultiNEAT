cmake_minimum_required(VERSION 3.15)
project(MultiNEAT VERSION 0.6.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MULTINEAT_BUILD_PYTHON "Build the Python extension module (needs pybind11)" ON)
option(MULTINEAT_BUILD_CPP_LIBRARY "Build the pure C++ static library" ON)
option(MULTINEAT_BUILD_EXAMPLES "Build the C++ XOR example / smoke test (needs the C++ library)" ON)

set(MULTINEAT_SOURCES
    src/ESHyperNEAT.cpp
    src/Genome.cpp
    src/Innovation.cpp
    src/NeuralNetwork.cpp
    src/Parameters.cpp
    src/PhenotypeBehavior.cpp
    src/Physics2D.cpp
    src/Population.cpp
    src/Random.cpp
    src/Species.cpp
    src/Substrate.cpp
    src/Traits.cpp
    src/Utils.cpp
)

if(MSVC)
    set(MULTINEAT_COMPILE_OPTIONS /W4 /bigobj /utf-8 /permissive-)
    set(MULTINEAT_COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
else()
    set(MULTINEAT_COMPILE_OPTIONS -Wall -Wextra)
    set(MULTINEAT_COMPILE_DEFINITIONS)
endif()

# The pure C++ library (no Python dependency)
if(MULTINEAT_BUILD_CPP_LIBRARY)
    add_library(multineat STATIC ${MULTINEAT_SOURCES})
    target_include_directories(multineat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(multineat PRIVATE ${MULTINEAT_COMPILE_OPTIONS})
    target_compile_definitions(multineat PUBLIC ${MULTINEAT_COMPILE_DEFINITIONS})
    set_target_properties(multineat PROPERTIES POSITION_INDEPENDENT_CODE ON)
    install(TARGETS multineat ARCHIVE DESTINATION lib)
    install(DIRECTORY src/ DESTINATION include/MultiNEAT FILES_MATCHING PATTERN "*.h")
endif()

if(MULTINEAT_BUILD_EXAMPLES AND MULTINEAT_BUILD_CPP_LIBRARY)
    add_executable(multineat_xor src/Main.cpp)
    target_link_libraries(multineat_xor PRIVATE multineat)
    target_compile_options(multineat_xor PRIVATE ${MULTINEAT_COMPILE_OPTIONS})

    enable_testing()
    add_test(NAME xor_cpp COMMAND multineat_xor 500 1)
endif()

# The Python extension module MultiNEAT._MultiNEAT
if(MULTINEAT_BUILD_PYTHON)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        # fall back to the pybind11 installed in the Python environment (pip install pybind11)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _pybind11_result)
        if(_pybind11_result EQUAL 0)
            find_package(pybind11 CONFIG REQUIRED PATHS "${_pybind11_dir}" NO_DEFAULT_PATH)
        else()
            message(FATAL_ERROR "pybind11 not found. Install it with: pip install pybind11")
        endif()
    endif()

    pybind11_add_module(_MultiNEAT src/PythonBindings.cpp ${MULTINEAT_SOURCES})
    target_include_directories(_MultiNEAT PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(_MultiNEAT PRIVATE PYTHON_BINDINGS ${MULTINEAT_COMPILE_DEFINITIONS})
    target_compile_options(_MultiNEAT PRIVATE ${MULTINEAT_COMPILE_OPTIONS})

    if(SKBUILD)
        # building a wheel with scikit-build-core
        install(TARGETS _MultiNEAT LIBRARY DESTINATION MultiNEAT)
    else()
        # in-tree build: put the module next to the Python package so "import MultiNEAT" works from the repo
        set_target_properties(_MultiNEAT PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/MultiNEAT
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/MultiNEAT
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/MultiNEAT
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/MultiNEAT)
    endif()
endif()
