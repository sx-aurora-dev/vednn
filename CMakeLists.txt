cmake_minimum_required(VERSION 3.0)
project(vednn)

option(USE_OPENMP   "Use OpenMP" ON)
option(BUILD_SHARED_LIB "Build Shared Library" OFF)
option(LOCAL_LLVM   "Try llvm+clang from .local? (use to avoid system clang, if nec.)" OFF)
set(USE_FTRACE CACHE STRING "ftrace+veperf support [0]|1|2, 1 instruments less than 2" )
set_property(CACHE USE_FTRACE PROPERTY STRINGS OFF 1 2) # informative only

# Can use env variables NCC and NCXX to override default VE compiler
if(DEFINED ENV{NCC})
    set(NCC ENV{NCC} CACHE FILEPATH "ncc command")
else()
    set(NCC "/opt/nec/ve/bin/ncc" CACHE FILEPATH "Path of ncc")
endif()
if(DEFINED ENV{NCXX})
    set(NCXX ENV{NCXX} CACHE FILEPATH "nc++ command")
else()
    set(NCXX "/opt/nec/ve/bin/nc++" CACHE FILEPATH "Path of nc++")
endif()
set(CMAKE_AR /opt/nec/ve/bin/nar)

# If unspecified build release with debug info
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
    # Informational:
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) # avoid default install --> /usr/local
	set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

######################################################

if(LOCAL_LLVM)
	# Tailor this if cmake still does not pick up your desired clang
	set(VEDNN_LLVM_HINT "~/.local")
	find_package(LLVM REQUIRED CONFIG
		# 0   : -DLLVM_DIR on cmake command line
		# 1,2 : CMAKE_PREFIX_PATH or CMAKE_PROGRAM_PATH (a: from cache, b: from env variable)
		# 3   : try a local path
		HINTS ${VEDNN_LLVM_HINT}
		# 4   : try a "system location"
		PATHS /opt/nec/ve/unsupported/llvm-ve /proj/share/local/x86_64/llvm
		#PATH_SUFFIXES lib/cmake/clang lib/cmake/llvm
		NO_SYSTEM_ENVIRONMENT_PATH
		)
else()
	find_package(LLVM REQUIRED CONFIG)
endif()
message(STATUS "LLVM_TOOLS_BINARY_DIR : ${LLVM_TOOLS_BINARY_DIR}")

if(NOT USE_FTRACE) # enforce 0 or 1 or 2 -- default build should prefer 0
    set(USE_FTRACE "0")
else() # specify USE_FTRACE 2 to instrument EVERY function, else just selected regions are ftraced
    if(USE_FTRACE STREQUAL "0")
    else()
        add_definitions(-DFTRACE) # lib hand-selects key ftrace_begin/end regions
        if(USE_FTRACE STREQUAL "1")
            # selected ftrace regions
        else()
            set(USE_FTRACE "2") # -ftrace compile (all funcs)
        endif()
    endif()
endif()
if(USE_OPENMP)
    add_definitions(-DVEDNN_USE_OPENMP)
endif()
message(STATUS "USE_FTRACE --------> ${USE_FTRACE}")
message(STATUS "USE_OPENMP --------> ${USE_OPENMP}")
message(STATUS "BUILD_SHARED ------> ${BUILD_SHARED}")
# ncc-2.x I think supplies ftrace.h and veperf.h, and enables VE_PROGINFO by default,
# so we no longer need a custom path and library for VEPERF header/library

# vednnx may link to blas libraries (for gen-dnn im2col in 'src/wrap/vconv')
set(NLCDIR "/opt/nec/ve/nlc/2.0.0" CACHE STRING "VE blas install directory")
# Some targets want finer control of optimization flags,
# so let defaults only add debug flags for now:
set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -DTMP11")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-g2 -DNDEBUG -DTMP12")
#set(CMAKE_C_FLAGS_MINSIZEREL "")
set(CMAKE_C_FLAGS_DEBUG "-g2 -DTMP10")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -DTMP13")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g2 -DNDEBUG -DTMP14")
#set(CMAKE_C_FLAGS_MINSIZEREL "")
set(CMAKE_CXX_FLAGS_DEBUG "-g2 -DTMP15")

message(STATUS "PROJECT_ROOT/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
add_subdirectory(src)
add_subdirectory(src/wrap)
add_subdirectory(src/C)
add_subdirectory(src/intrinsic)
# vim: et ts=4 sw=4 ai
