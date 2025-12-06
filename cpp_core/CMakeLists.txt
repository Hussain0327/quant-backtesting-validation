cmake_minimum_required(VERSION 3.15)
project(qss_core VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Find pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    # Try to find it via pip/conda installation
    execute_process(
        COMMAND python3 -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
    else()
        message(STATUS "pybind11 not found, Python bindings will not be built")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(QSS_SOURCES
    src/random_generator.cpp
    src/portfolio.cpp
    src/monte_carlo.cpp
    src/statistics.cpp
)

# Header files
set(QSS_HEADERS
    include/random_generator.hpp
    include/portfolio.hpp
    include/monte_carlo.hpp
    include/statistics.hpp
)

# Static library
add_library(qss_static STATIC ${QSS_SOURCES})
target_include_directories(qss_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Shared library
add_library(qss_shared SHARED ${QSS_SOURCES})
target_include_directories(qss_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(qss_shared PROPERTIES OUTPUT_NAME qss)

# Python bindings (if pybind11 found)
if(pybind11_FOUND)
    pybind11_add_module(qss_core src/bindings.cpp ${QSS_SOURCES})
    target_include_directories(qss_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Install Python module
    install(TARGETS qss_core
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python)
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(qss_static PRIVATE Threads::Threads)
target_link_libraries(qss_shared PRIVATE Threads::Threads)
if(pybind11_FOUND)
    target_link_libraries(qss_core PRIVATE Threads::Threads)
endif()

# Install targets
install(TARGETS qss_static qss_shared
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES ${QSS_HEADERS} DESTINATION include/qss)

# Tests (optional)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()

    # Find or fetch GoogleTest
    find_package(GTest CONFIG)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.12.1
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    # Test executable
    add_executable(qss_tests
        tests/test_random.cpp
        tests/test_portfolio.cpp
        tests/test_monte_carlo.cpp
        tests/test_statistics.cpp
    )
    target_link_libraries(qss_tests PRIVATE qss_static GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(qss_tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== QSS Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Python bindings: ${pybind11_FOUND}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "=========================")
message(STATUS "")
