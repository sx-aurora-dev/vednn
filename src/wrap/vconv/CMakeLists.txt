cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O4 -finline -finline-functions -DNDEBUG -DCMK_VCONV")
#set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O4 -finline -finline-functions -DNDEBUG -DCMK_VCONV")

if(USE_FTRACE AND USE_FTRACE STREQUAL 2)
    #message(STATUS " ??? adding ftrace compilation flag ??? ")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -ftrace -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftrace -pthread")
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
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DCMK_VCONV")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMK_VCONV")
message(STATUS "src/wrap/vconv CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
message(STATUS "src/wrap/vconv CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "NLCDIR is ${NLCDIR}")

if(NOT IS_DIRECTORY "${NLCDIR}") # usually set in some higher CMakeLists.txt ...
    set(NLCDIR "/opt/nec/ve/nlc/2.0.0" CACHE STRING "VE blas install directory")
endif()

# I now have local changes (for ncc 3.0 'restrict')
# unpack has to run at **cmake** time. add_custom_target...i
#        PRE_BUILD is not quite good enough.
#if(( (NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/)
#    OR (NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
#    OR (NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/)
#    ) AND EXISTS ${PROJECT_SOURCE_DIR}/vconv-ve.tar.gz)
#    message(STATUS "Unpacking vconv-ve.tar.gz distribution")
#    execute_process(
#        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/include/
#        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/lib/
#        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/
#        COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_SOURCE_DIR}/vconv-ve.tar.gz
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#        )
#endif()
file(GLOB_RECURSE VGEMM_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/detail/vgemm/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/detail/vgemm/*.hpp
    )
file(GLOB_RECURSE VGEMM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vgemm/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vgemm/*.cpp
    )
file(GLOB_RECURSE VCONV_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/detail/vconv/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/detail/vconv/*.hpp
    )
file(GLOB_RECURSE VCONV_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vconv/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vconv/*.cpp
    )
file(GLOB WRAP_VCONV_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    )
file(GLOB WRAP_VCONV_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )
message(STATUS "vgemm headers ${VGEMM_HEADERS}")
message(STATUS "vgemm sources ${VGEMM_SOURCES}")
message(STATUS "vconv headers ${VCONV_HEADERS}")
message(STATUS "vconv sources ${VCONV_SOURCES}")
message(STATUS "wrap_vconv headers ${VGEMM_HEADERS}")
message(STATUS "wrap_vconv sources ${VGEMM_SOURCES}")
file(COPY ${WRAP_VCONV_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/wrap/)
file(COPY ${VGEMM_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/wrap/detail/vgemm)
file(COPY ${VCONV_HEADERS} DESTINATION ${CMAKE_BINARY_DIR}/include/wrap/detail/vconv)
# wrap_vconv headers installed at CMAKE_INSTALL_PREFIX/include/wrap/

#include_directories("../wrap" "../.." "../../C" "./include/vgemm" "./include/gendnn/vconv" "${NLCDIR}/include")
#include_directories("../wrap" "../.." "../../C" "${CMAKE_BINARY_DIR}/include/" "${NLCDIR}/include")
include_directories(
    #"../wrap"
    "."
    "../.."
    "../../C"
    #"${CMAKE_BINARY_DIR}/include"
    "${CMAKE_BINARY_DIR}/include/wrap"
    "${CMAKE_BINARY_DIR}/include/wrap/detail/vgemm/"
    "${CMAKE_BINARY_DIR}/include/wrap/detail/vconv"
    "${NLCDIR}/include")

# The Makefile has the best set of compilation flags
add_library(vconv_code OBJECT
    direct_gendnn.cpp
    md_util.cpp
    ${VGEMM_SOURCES}
    ${VCONV_SOURCES}
    ${WRAP_VCONV_SOURCES}
    )
# Oh, need to run vconv_unPack at CMAKE-TIME [not build-time] to get file globs correct
#add_dependencies(vconv_code vconv_unPack)

#message(STATUS "src/C/vconv/ LIBWRAP ${LIBWRAP}")
#target_link_libraries(vconv_code cblas blas)

# libvednnx iterator exposes low-level calls (with different parameter ordering)
# Some external projects may want to directly access the low-level libvednn
# impls, with their slightly different parameter ordering... for now.
# Normally the user will never include these (even indirectly).
# As libvednnx.h matures it may be possible to remove these headers.
install(FILES
    vednn2gendnn.h
    md_util.h
    ${WRAP_VCONV_HEADERS}
    DESTINATION include/wrap
    )
# vim: et ts=4 sw=4 ai
