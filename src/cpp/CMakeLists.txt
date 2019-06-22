cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp-xgboost-test LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # -std=c++11 instead of -std=gnu++11

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(xgboost CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(PREDICT "predict")

add_executable(${PREDICT} ${CMAKE_CURRENT_SOURCE_DIR}/src/predict.cpp)
target_include_directories(${PREDICT} SYSTEM PRIVATE ${XGBOOST_INCLUDE_DIRS})
target_include_directories(${PREDICT} SYSTEM PRIVATE ${CXXOPTS_ROOT}/include/)
target_link_libraries(
    ${PREDICT}
    xgboost::xgboost
    Threads::Threads
)

if(MSVC)
    target_compile_options(${PREDICT} PRIVATE /W4)
else()
    target_compile_options(${PREDICT} PRIVATE -Wall -Wextra)
endif()

