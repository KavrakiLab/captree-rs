cmake_minimum_required(VERSION 3.1...3.22)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
project(
  vamp
  VERSION 0.1
  LANGUAGES C CXX
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FORCE_COLORED_OUTPUT "Always produce ANSI-colored output." ON)
if(FORCE_COLORED_OUTPUT)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  set(arch "-mcpu=native -mtune=native")
  set(FAST_ARGS "-fno-honor-infinities -fapprox-func -fno-math-errno -fassociative-math -fno-signed-zeros -fno-trapping-math -fno-rounding-math -ffp-contract=fast")
else()
  set(arch "-march=native")
  set(FAST_ARGS "-fno-math-errno -ffinite-math-only -fassociative-math -fno-signed-zeros -fno-trapping-math -fno-rounding-math -ffp-contract=fast")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${arch} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g -mno-avx512f")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -flto ${FAST_ARGS}")

message(STATUS "Building ${CMAKE_BUILD_TYPE}")

find_package(Eigen3 REQUIRED NO_MODULE)

set(HIGHFIVE_USE_BOOST Off)
set(HIGHFIVE_USE_INSTALL_DEPS On)
find_package(HighFive REQUIRED)

set(HILBERT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/hilbert-sort/hilbert-sort/include/)
set(MORTON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/hilbert-sort/external/libmorton/libmorton/include/)
set(PDQSORT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/pdqsort)

add_executable(hilbert_test hilbert_test.cc)
target_include_directories(hilbert_test SYSTEM PRIVATE ${HILBERT_INCLUDE_DIRS} ${MORTON_INCLUDE_DIRS} ${PDQSORT_INCLUDE_DIRS})
target_link_libraries(hilbert_test PRIVATE HighFive Eigen3::Eigen)
