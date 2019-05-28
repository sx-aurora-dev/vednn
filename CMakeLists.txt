cmake_minimum_required(VERSION 3.0)

set(NCC "/opt/nec/ve/bin/ncc" CACHE FILEPATH "Path of ncc")
set(NCXX "/opt/nec/ve/bin/nc++" CACHE FILEPATH "Path of nc++")

option(USE_OPENMP   "Use OpenMP" ON)
option(BUILD_SHARED "Build Shared Library" OFF)
option(LOCAL_LLVM   "Try llvm+clang from .local? (use to avoid system clang, if nec.)" OFF)
set(USE_FTRACE CACHE STRING "ftrace+veperf support [0]|1|2, 1 instruments less than 2" )
set_property(CACHE USE_FTRACE PROPERTY STRINGS OFF 1 2) # informative only

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) # avoid default install --> /usr/local
	set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

if(LOCAL_LLVM)
	# Tailor this if cmake still does not pick up your desired clang
	set(VEDNN_LLVM_HINT "~/.local")
	find_package(LLVM REQUIRED CONFIG
		# 0   : -DLLVM_DIR on cmake command line
		# 1,2 : CMAKE_PREFIX_PATH or CMAKE_PROGRAM_PATH (a: from cache, b: from env variable)
		# 3   : try a local path
		HINTS ${VEDNN_LLVM_HINT}
		# 4   : try a "system location"
		PATHS /proj/share/local/x86_64/llvm
		#PATH_SUFFIXES lib/cmake/clang lib/cmake/llvm
		NO_SYSTEM_ENVIRONMENT_PATH
		)
else()
	find_package(LLVM REQUIRED CONFIG)
endif()
message(STATUS "LLVM_TOOLS_BINARY_DIR : ${LLVM_TOOLS_BINARY_DIR}")


add_subdirectory(src/intrinsic)
add_subdirectory(src/C)
add_subdirectory(src)

