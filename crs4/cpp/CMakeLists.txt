# inspired by
# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/custom_operations/custom_operator/create_a_custom_operator.html

cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

project(crs4_cassandra_plugin LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")

# TODO(klecki): When the test container gets a CMake that supports C++20 as a proper option,
# swap those lines
# set(CMAKE_CUDA_STANDARD 20)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

include_directories(SYSTEM "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# Required system libraries (OpenSSL dependencies of the Cassandra driver)
find_library(SSL_LIB ssl REQUIRED)
find_library(CRYPTO_LIB crypto REQUIRED)

include(FetchContent)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# Enable Position Independent Code (PIC) for the static libraries
# Required when linking statically into a shared library (.so)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build libuv from source so pip install does not require a system libuv package.
FetchContent_Declare(
    libuv
    GIT_REPOSITORY https://github.com/libuv/libuv.git
    GIT_TAG        v1.52.1
    GIT_SHALLOW    TRUE
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LIBUV_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(LIBUV_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(LIBUV_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LIBUV_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libuv)

if(TARGET uv_a)
    set(UV_TARGET uv_a)
elseif(TARGET uv)
    set(UV_TARGET uv)
else()
    message(FATAL_ERROR "Failed to build libuv: no library target was generated")
endif()

# Fetch and build cassandra driver statically from source
# This is required for distributing the plugin as a wheel, as users
# generally do not have the Cassandra C++ driver installed.
FetchContent_Declare(
    cassandra_driver
    GIT_REPOSITORY https://github.com/datastax/cpp-driver.git
    GIT_TAG        2.17.0
    GIT_SHALLOW    TRUE
)
set(CASS_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(CASS_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(CASS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CASS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# cpp-driver 2.17.0 is not C++20 compatible (uses std::memory_order as unscoped enum)
# Build it with C++17 instead
set(SAVED_CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD 17)

# Enable Position Independent Code (PIC) for the static library
# Required when linking statically into a shared library (.so)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_MakeAvailable(cassandra_driver)
set(CMAKE_CXX_STANDARD ${SAVED_CMAKE_CXX_STANDARD})

set(CASSANDRA_TARGET cassandra_static)
set(CASSANDRA_INCLUDE_DIR ${cassandra_driver_SOURCE_DIR}/include)

execute_process(
        COMMAND python3 -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
        OUTPUT_VARIABLE DALI_LIB_DIR)
string(STRIP ${DALI_LIB_DIR} DALI_LIB_DIR)

execute_process(
        COMMAND python3 -c "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
string(STRIP ${DALI_COMPILE_FLAGS} DALI_COMPILE_FLAGS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS} ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${DALI_COMPILE_FLAGS} ")
link_directories("${DALI_LIB_DIR}")
link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")  # -lcudart

add_library(crs4cassandra SHARED cassandra_dali_interactive.cc cassandra_dali_selffeed.cc cassandra_dali_decoupled.cc batch_loader.cc)

target_include_directories(crs4cassandra PRIVATE ${CASSANDRA_INCLUDE_DIR})
# Link against the static cassandra target and explicit system dependencies
target_link_libraries(crs4cassandra dali cudart ${CASSANDRA_TARGET} ${UV_TARGET} ${SSL_LIB} ${CRYPTO_LIB})
