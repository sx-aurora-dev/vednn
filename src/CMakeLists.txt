cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
set(CMAKE_CXX_COMPILER /opt/nec/ve/bin/nc++)

#set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O4")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O4")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -fcse-after-vectorization -msched-block")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fcse-after-vectorization -msched-block")
message(STATUS "src/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")

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
set(NLCDIR "/opt/nec/ve/nlc/2.0.0")
set(CMAKE_CFLAGS     "${CMAKE_CFLAGS} -I${NLCDIR}/include")
set(CMAKE_CXXFLAGS "${CMAKE_CXXFLAGS} -I${NLCDIR}/include")
#set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH} ${NLCDIR}/lib")
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH} ${NLCDIR}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${NLCDIR}/lib")

add_library(${LIBNAME} ${LIB_MODE}
        $<TARGET_OBJECTS:vconv_code>
        $<TARGET_OBJECTS:vednn_c_code>
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
add_dependencies(${LIBNAME} unPack)
if(BUILD_SHARED)
    set_target_properties(${LIBNAME} PROPERTIES
        INSTALL_RPATH "\$ORIGIN/../lib"
        )
endif()

add_library(${LIBWRAP} ${LIB_MODE}
        $<TARGET_OBJECTS:vednn_wrap>)

if(USE_FTRACE)
	target_link_libraries(${LIBNAME} veperf)
	target_link_libraries(${LIBWRAP} ${LIBNAME} veperf)
endif()

if(USE_OPENMP)
    #target_link_libraries(${LIBNAME} ${NLCDIR}/libcblas ${NLCDIR}/libblas_openmp.so)
    target_link_libraries(${LIBNAME} -L${NLCDIR}/lib
        -lcblas -lblas_openmp)
else()
    target_link_libraries(${LIBNAME}
        ${NLCDIR}/lib/libcblas.so ${NLCDIR}/lib/libblas_sequential.so)
endif()

install(TARGETS ${LIBNAME} ${LIBWRAP} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES vednn.h DESTINATION include)
