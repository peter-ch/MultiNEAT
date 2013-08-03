if (${CMAKE_VERBOSE_MAKEFILE})
	set(NODE_GYP_VERBOSE "-v")
	set(NPM_VERBOSE "--loglevel verbose")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
	set(NODE_GYP_DEBUG "-d")
	set(NODE_GYP_BUILD_TYPE "Debug")
else ()
	set(NODE_GYP_BUILD_TYPE "Release")
endif()

if (NOT NODE_GYP_EXECUTABLE)
	set(NODE_GYP_EXECUTABLE "node-gyp")
endif ()

if (NOT NPM_EXECUTABLE)
	set(NPM_EXECUTABLE "npm")
endif ()

if (ARCH STREQUAL "i386")
	set(NPM_CONFIG_ARCH "npm_config_arch=ia32")
elseif (ARCH STREQUAL "x86_64") 
	set(NPM_CONFIG_ARCH "npm_config_arch=x64")
endif ()
		
if (NODE_GYP_TARGET)
	set(NODE_GYP_TARGET_PARAM "--target=${NODE_GYP_TARGET}")
endif ()

set(NODE_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/module")

# file(MAKE_DIRECTORY ${directory})
configure_file(binding.gyp.in "${NODE_MODULE_DIR}/binding.gyp")
add_custom_command(OUTPUT "${NODE_MODULE_DIR}/build/Makefile"
                   COMMAND ${NPM_CONFIG_ARCH} "${NODE_GYP_EXECUTABLE}" configure ${NODE_GYP_TARGET_PARAM} ${NODE_GYP_DEBUG} ${NODE_GYP_VERBOSE}
                   WORKING_DIRECTORY "${NODE_MODULE_DIR}"
                   DEPENDS binding.gyp.in)
                   
add_custom_command(OUTPUT "${NODE_MODULE_DIR}/build/${NODE_GYP_BUILD_TYPE}/MultiNEAT.node"
				   COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/src" "${NODE_MODULE_DIR}/src"
				   COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/package.json" "${NODE_MODULE_DIR}" 
                   COMMAND "${NODE_GYP_EXECUTABLE}" build ${NODE_GYP_VERBOSE}
                   WORKING_DIRECTORY "${NODE_MODULE_DIR}"
                   DEPENDS MultiNEAT src/NodeBindings.cpp src/NodeBindings.h)

add_custom_target(MultiNEAT_node ALL DEPENDS "${NODE_MODULE_DIR}/build/Makefile" "${NODE_MODULE_DIR}/build/${NODE_GYP_BUILD_TYPE}/MultiNEAT.node")

install(CODE "execute_process(COMMAND ${NPM_EXECUTABLE} install -g ${NPM_VERBOSE} \"${NODE_MODULE_DIR}\")")