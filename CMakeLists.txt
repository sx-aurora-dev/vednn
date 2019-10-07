cmake_minimum_required(VERSION 3.0)

if(DEFINED ENV{NCC})
    set(NCC ENV{NCC} CACHE FILEPATH "ncc command")
    # TODO set(NCXX ENV{NCC} CACHE FILEPATH "nc++ command")
else()
    set(NCC "/opt/nec/ve/bin/ncc" CACHE FILEPATH "Path of ncc")
    set(NCXX "/opt/nec/ve/bin/nc++" CACHE FILEPATH "Path of nc++")
endif()

option(USE_OPENMP   "Use OpenMP" ON)
option(BUILD_SHARED "Build Shared Library" OFF)
option(LOCAL_LLVM   "Try llvm+clang from .local? (use to avoid system clang, if nec.)" OFF)
set(USE_FTRACE CACHE STRING "ftrace+veperf support [0]|1|2, 1 instruments less than 2" )
set_property(CACHE USE_FTRACE PROPERTY STRINGS OFF 1 2) # informative only

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

if(NOT USE_FTRACE) # enforce 0 or 1 or 2
    set(USE_FTRACE "1")
else() # specify USE_FTRACE 2 to instrument EVERY function, else just selected regions are ftraced
    if(USE_FTRACE STREQUAL "0")
    else()
        if(USE_FTRACE STREQUAL "2")
        else()
            set(USE_FTRACE "1")
        endif()
    endif()
endif()
message(STATUS "USE_FTRACE --------> ${USE_FTRACE}")
if(USE_FTRACE)
	# ftrace/veperf location
	find_file(found_VEPERF_H NAME veperf.h
	    NO_DEFAULT_PATH
        # /opt/nec/ve default path ?
	    PATHS /usr/uhome/aurora/mpc/pub/veperf/latest
	    PATH_SUFFIXES include
	    )
	message(STATUS "veperf.h --> ${found_VEPERF_H}")
	if(NOT found_VEPERF_H)
	    message(WARNING "ve.cmake: veperf.h not found (looking for veperf.h [ftrace.h might be there too])")
    else()
        get_filename_component(VE_VEPERF_INCLUDE "${found_VEPERF_H}" DIRECTORY)
        get_filename_component(VE_VEPERF_DIR "${VE_VEPERF_INCLUDE}" DIRECTORY)
        set(VE_VEPERF_DIR "${VE_VEPERF_DIR}" CACHE PATH "Aurora ftrace/veperf root directory" FORCE)
        set(VE_VEPERF_INCLUDE "${VE_VEPERF_INCLUDE}" CACHE PATH "Aurora ftrace/veperf include path" FORCE)
        set(VE_VEPERF_INCFLAGS "-I${VE_VEPERF_INCLUDE}" CACHE STRING "Aurora ftrace/veperf include path" FORCE)
        set(VE_VEPERF_LDFLAGS "-L${VE_VEPERF_DIR}/lib" CACHE STRING "Aurora ftrace/veperf C/CXX compile/link options" FORCE)
        set(VE_VEPERF_LIBS "-lveperf") # NOTE: there are static and shared libs
        mark_as_advanced(VE_VEPERF_DIR VE_VEPERF_INCLUDE VE_VEPERF_LDFLAGS VE_VEPERF_LIBS)
        # Q: What is difference between   libveperf and libveperf_sp
        # TODO: add library existence and compilation checks
        message(STATUS "veperf.h                         : ${found_VEPERF_H}")
        message(STATUS "VE_VEPERF_INCLUDE                : ${VE_VEPERF_INCLUDE}")
        message(STATUS "VE_VEPERF_DIR [ftrace|veperf]    : ${VE_VEPERF_DIR}")
        message(STATUS "VE_VEPERF_INCFLAGS               : ${VE_VEPERF_INCFLAGS}")
        message(STATUS "VE_VEPERF_LDFLAGS                : ${VE_VEPERF_LDFLAGS}")
        message(STATUS "VE_VEPERF_LIBS                   : ${VE_VEPERF_LIBS}")
        # Expected libraries: libveperf.{a|so} headers: ftrace.h veperf.h
        # add LDFLAGS="-lveperf" (nothing for ftrace)
        # VE_VEPERF_DIR for ftrace/veperf
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -DFTRACE=${USE_FTRACE} -I${VE_VEPERF_INCLUDE}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFTRACE=${USE_FTRACE} -I${VE_VEPERF_INCLUDE}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,${VEPERF_LIB_DIR}")
    endif()
    find_library(veperf NAMES veperf.a PATHS ${VE_VEPERF_DIR}/lib NO_DEFAULT_PATH)
endif()

message(STATUS "PROJECT_ROOT/ CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
add_subdirectory(src)
add_subdirectory(src/intrinsic)
add_subdirectory(src/C)
add_subdirectory(src/wrap)
# vim: et ts=4 sw=4 ai
