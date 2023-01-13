cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(xgboost-example LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

include(FetchContent)

find_package(cxxopts CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Build static version of xgboost library
set(BUILD_STATIC_LIB ON)
FetchContent_Declare(
    xgboost
    SYSTEM
    GIT_REPOSITORY https://github.com/dmlc/xgboost.git
    GIT_TAG v1.7.3
)
FetchContent_MakeAvailable(xgboost)

set(PREDICT "predict")

add_executable(${PREDICT} ${CMAKE_CURRENT_SOURCE_DIR}/src/predict.cpp)

set_target_properties(${PREDICT} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
target_link_libraries(${PREDICT} PRIVATE
    xgboost
    cxxopts::cxxopts
    Threads::Threads
)

if(MSVC)
    target_compile_options(${PREDICT} PRIVATE /W4)
else()
    target_compile_options(${PREDICT} PRIVATE -W -Wall -Wextra)
endif()
