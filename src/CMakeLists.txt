cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
set(CMAKE_CXX_COMPILER /opt/nec/ve/bin/nc++)

# vednn   : core vednn library
# vednnx  : extensions for tests (is impl ok? iteration over ok impls)
if(USE_OPENMP)
	set(LIBNAME vednn_openmp)
	set(LIBWRAP vednnx_openmp)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fopenmp")
else()
	set(LIBNAME vednn_sequential)
	set(LIBWRAP vednnx_sequential)
endif()

if(BUILD_SHARED_LIB)
	set(LIB_MODE SHARED)
else()
	set(LIB_MODE STATIC)
endif()

# for vconv (does it need to be here?) XXX
set(CMAKE_CFLAGS     "${CMAKE_CFLAGS} -I${NLCDIR}/include")
set(CMAKE_CXXFLAGS "${CMAKE_CXXFLAGS} -I${NLCDIR}/include")
#set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH} ${NLCDIR}/lib")
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH} ${NLCDIR}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${NLCDIR}/lib")

message(STATUS "src/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")

add_library(${LIBNAME} ${LIB_MODE}
        $<TARGET_OBJECTS:vconv_code>
        $<TARGET_OBJECTS:vednn_c_code>
        $<TARGET_OBJECTS:vednn_gemm_convolution>
        $<TARGET_OBJECTS:vednn_intrinsic_convolution_forward>
        $<TARGET_OBJECTS:vednn_intrinsic_convolution_forward_addbias>
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

add_library(${LIBWRAP} ${LIB_MODE}
        $<TARGET_OBJECTS:vednn_wrap>)

#if(USE_FTRACE)
#    if(BUILD_SHARED_LIB)
#        target_link_libraries(${LIBNAME} -L${VE_VEPERF_DIR}/lib veperf)
#        target_link_libraries(${LIBWRAP} -L${VE_VEPERF_DIR}/lib veperf)
#        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${VE_VEPERF_DIR}/lib")
#    else()
#        target_link_libraries(${LIBNAME} ${veperf})
#        target_link_libraries(${LIBWRAP} ${veperf})
#    endif()
#endif()

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
install(FILES vednn.h DESTINATION include)
