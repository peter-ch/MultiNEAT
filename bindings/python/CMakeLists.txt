set(Boost_USE_STATIC_LIBS OFF)
find_package(Boost COMPONENTS python REQUIRED)

if (${CMAKE_VERBOSE_MAKEFILE})
	set(DISTUTILS_VERBOSE "-v")
endif()

find_package(PythonInterp REQUIRED)

configure_file(setup.py.in "${CMAKE_CURRENT_BINARY_DIR}/setup.py")

add_custom_command(OUTPUT lib
                   COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/setup.py" build_ext -b "${CMAKE_CURRENT_BINARY_DIR}/lib" -t "${CMAKE_CURRENT_BINARY_DIR}/temp"  ${DISTUTILS_VERBOSE}
                   COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/setup.py" build_py -d "${CMAKE_CURRENT_BINARY_DIR}/lib"  ${DISTUTILS_VERBOSE}
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                   DEPENDS setup.py.in MultiNEAT.py src/PythonBindings.cpp src/PythonBindings.h)

add_custom_target(MultiNEAT_python ALL DEPENDS MultiNEAT lib)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} \"${CMAKE_CURRENT_BINARY_DIR}/setup.py\" install_lib --skip-build -b \"${CMAKE_CURRENT_BINARY_DIR}/lib\" WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\")")
