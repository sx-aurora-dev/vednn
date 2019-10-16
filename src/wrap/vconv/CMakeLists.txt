cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
if(DEFINED ENV{NCC})
    set(CMAKE_C_COMPILER ENV{NCC})
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O4 -finline -finline-functions -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O4 -finline -finline-functions -DNDEBUG")

if(USE_FTRACE AND NOT USE_FTRACE STREQUAL 1)
    message(STATUS " ??? adding ftrace compilation flag ??? ")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -ftrace")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftrace")
endif()

if(USE_OPENMP)
    add_definitions(-DVEDNN_USE_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if(BUILD_SHARED_LIB)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

message(STATUS "src/wrap/vconv CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
message(STATUS "src/wrap/vconv CMAKE_CXX_FLAGS=${CMAKE_C_FLAGS}")
message(STATUS "NLCDIR is ${NLCDIR}")

if(NOT IS_DIRECTORY "${NLCDIR}") # usually set in some higher CMakeLists.txt ...
    set(NLCDIR "/opt/nec/ve/nlc/2.0.0" CACHE STRING "VE blas install directory")
endif()

#add_custom_target(vconv_unPack ALL)
#add_custom_command(TARGET vconv_unPack PRE_BUILD
#    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/include/
#    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/lib/
#    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/
#    COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/vconv-ve.tar.gz
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#    DEPENDS ${PROJECT_SOURCE_DIR}/vconv-ve.tar.gz
#    COMMENT "Unpacking vconv-ve.tar.gz into ${CMAKE_SOURCE_DIR}"
#    VERBATIM)
if(( (NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    OR (NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
    OR (NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/)
    ) AND EXISTS ${PROJECT_SOURCE_DIR}/vconv-ve.tar.gz)
    message(STATUS "Unpacking vconv-ve.tar.gz distribution")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/include/
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/lib/
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/vconv-ve.tar.gz
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
endif()
file(GLOB_RECURSE VCONV_HEADERS
    ${CMAKE_BINARY_DIR}/include/*.h
    ${CMAKE_BINARY_DIR}/include/*.hpp
    )
file(GLOB_RECURSE VCONV_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    )
message(STATUS "vconv headers ${VCONV_HEADERS}")
message(STATUS "vconv sources ${VCONV_SOURCES}")

include_directories("../wrap" "../.." "../../C" "./include/gen-dnn/vconv" "./include/gen-dnn/vgemm" "${NLCDIR}/include")

# The Makefile has the best set of compilation flags
add_library(vconv_code OBJECT
    direct_gendnn.cpp
    md_util.cpp
    ${VCONV_SOURCES}
    )
# Oh, need to run vconv_unPack at CMAKE-TIME [not build-time] to get file globs correct
#add_dependencies(vconv_code vconv_unPack)

message(STATUS "src/C/vconv/ LIBWRAP ${LIBWRAP}")
#target_link_libraries(vconv_code cblas blas)

# libvednnx iterator exposes low-level calls (with different parameter ordering)
# Some external projects may want to directly access the low-level libvednn
# impls, with their slightly different parameter ordering... for now.
# Normally the user will never include these (even indirectly).
# As libvednnx.h matures it may be possible to remove these headers.
install(FILES
    vednn2gendnn.h
    md_util.h
    ${VCONV_HEADERS}
    DESTINATION include/C/vconv
    )
# vim: et ts=4 sw=4 ai