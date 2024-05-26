cmake_minimum_required(VERSION 3.5.0)
project(cpp-object-detection VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(CTest)

include(FetchContent)

###########################
# tensorflow lite API
###########################
FetchContent_Declare(
    tensorflow
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_PROGRESS TRUE
    GIT_TAG v2.16.1
    SOURCE_SUBDIR tensorflow/lite
)
FetchContent_MakeAvailable(tensorflow)

###############################
# cpp-object-detection lib
###############################
add_library(${PROJECT_NAME}-lib STATIC src/object_detection.cpp)

target_link_libraries(${PROJECT_NAME}-lib PRIVATE
    tensorflow-lite
)

target_include_directories(${PROJECT_NAME}-lib
    PRIVATE
    include
)

###################################
# cpp-object-detection executable
###################################
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
    tensorflow-lite
    ${PROJECT_NAME}-lib
)

###################################
# Testing and Coverage
###################################
option(BUILD_TESTS "Build tests" ON)
if (BUILD_TESTS)
    if (NOT CMAKE_BUILD_TYPE STREQUAL Debug)
        message(WARNING "Must set CMAKE_BUILD_TYPE to Debug in order to build tests! Tests will not be built.")
    else()
        enable_testing()
        add_subdirectory(test)

        target_compile_options(${PROJECT_NAME}-lib PRIVATE --coverage -O0)
        target_link_libraries(${PROJECT_NAME}-lib PRIVATE --coverage)
    endif()
endif()

###############################
# CPack
###############################
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
