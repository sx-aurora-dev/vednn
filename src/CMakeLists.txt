cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

#set(CMAKE_C_FLAGS "-O4")
#set(CMAKE_CXX_FLAGS "-O4")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -fcse-after-vectorization -msched-block")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fcse-after-vectorization -msched-block")
message(STATUS "src/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")

if(USE_OPENMP)
	set(LIBNAME vednn_openmp)
	set(LIBWRAP vednnx_openmp)
else()
	set(LIBNAME vednn_sequential)
	set(LIBWRAP vednnx_sequential)
endif()

if(BUILD_SHARED_LIB)
	set(LIB_MODE SHARED)
else()
	set(LIB_MODE STATIC)
endif()

add_library(${LIBNAME} ${LIB_MODE}
    $<TARGET_OBJECTS:vednn_c_code>
    $<TARGET_OBJECTS:vednn_intrinsic_convolution_forward>
    $<TARGET_OBJECTS:vednn_intrinsic_convolution_backward_data>
    $<TARGET_OBJECTS:vednn_intrinsic_convolution_backward_filter>
    $<TARGET_OBJECTS:vednn_intrinsic_linear_forward>
    $<TARGET_OBJECTS:vednn_intrinsic_linear_backward_data>
    $<TARGET_OBJECTS:vednn_intrinsic_linear_backward_weight>        
    $<TARGET_OBJECTS:vednn_intrinsic_maxpooling_backward>
    $<TARGET_OBJECTS:vednn_intrinsic_maxpooling_forward>
    $<TARGET_OBJECTS:vednn_intrinsic_activation>
    )
if(BUILD_SHARED)
    set_target_properties(${LIBNAME} PROPERTIES
        INSTALL_RPATH "\$ORIGIN/../lib"
        )
endif()

# vednnx extensions: iterator and jit api, gen-dnn subset
add_library(${LIBWRAP} ${LIB_MODE}
    $<TARGET_OBJECTS:vednn_wrap>
    $<TARGET_OBJECTS:vconv_code>
    )

if(USE_FTRACE)
	target_link_libraries(${LIBNAME} veperf)
	target_link_libraries(${LIBWRAP} ${LIBNAME} veperf)
endif()
if(USE_OPENMP)
    #target_link_libraries(${LIBNAME} ${NLCDIR}/libcblas ${NLCDIR}/libblas_openmp.so)
    target_link_libraries(${LIBWRAP} -L${NLCDIR}
        -lcblas -lcblas_openmp)
else()
    target_link_libraries(${LIBWRAP}
        ${NLCDIR}/libcblas.so ${NLCDIR}/libblas_sequential.so)
endif()
#

install(TARGETS ${LIBNAME} ${LIBWRAP} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES
    vednn.h vednn_util.hpp
    DESTINATION include)
# vim: sw=4 ts=4 et ai
