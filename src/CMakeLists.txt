cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

set(CMAKE_C_FLAGS "-O4")
set(CMAKE_CXX_FLAGS "-O4")

if(USE_OPENMP)
	set(LIBNAME vednn_openmp)
else()
	set(LIBNAME vednn_sequential)
endif()

if(BUILD_SHARED_LIB)
	set(LIB_MODE SHARED)
else()
	set(LIB_MODE STATIC)
endif()

add_library(${LIBNAME} ${LIB_MODE}
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

install(TARGETS ${LIBNAME} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(FILES vednn.h DESTINATION include)
