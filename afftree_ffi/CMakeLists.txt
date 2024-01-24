if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build)
    set(TARGET_DIR "debug")
else ()
    set(CARGO_CMD cargo build --release)
    set(TARGET_DIR "release")
endif ()

if(ENABLE_LTO)
    set(RUST_FLAGS "-Clinker-plugin-lto" "-Clinker=clang-17" "-Clink-arg=-fuse-ld=lld-17")
endif()

set(AFFTREE_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/libafftree_ffi.a")

set(AFFTREE_CXX "${CMAKE_CURRENT_BINARY_DIR}/afftree.cc")
add_library(afftree STATIC ${AFFTREE_CXX})
add_custom_command(
    OUTPUT ${AFFTREE_CXX}
    COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} RUSTFLAGS="${RUST_FLAGS}" ${CARGO_CMD}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/afftree_ffi/src/lib.rs.cc ${AFFTREE_CXX}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/afftree_ffi/src/lib.rs.h ${CMAKE_CURRENT_BINARY_DIR}/afftree.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(afftree pthread dl ${AFFTREE_LIB})