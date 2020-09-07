cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(xgboost-example-superbuild LANGUAGES CXX C VERSION 0.1.0)

include(ExternalProject)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Git)
if(NOT Git_FOUND)
    message(FATAL_ERROR "Git NOT found")
endif()

set(XGBOOST_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/xgboost)
set(XGBOOST_INSTALL_PREFIX ${XGBOOST_PREFIX}/install)
ExternalProject_Add(
    xgboost
    PREFIX ${XGBOOST_PREFIX}
    GIT_REPOSITORY "https://github.com/dmlc/xgboost.git"
    # GIT_TAG "v1.2.0"
    GIT_SHALLOW ON
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DBUILD_STATIC_LIB=ON
        -DCMAKE_INSTALL_PREFIX=${XGBOOST_INSTALL_PREFIX}
    LOG_INSTALL ON
)

set(CXXOPTS_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/third_party/cxxopts)
set(CXXOPTS_INSTALL_PREFIX ${CXXOPTS_PREFIX}/install)
ExternalProject_Add(
    cxxopts
    PREFIX ${CXXOPTS_PREFIX}
    URL "https://github.com/jarro2783/cxxopts/archive/v2.1.2.tar.gz"
    URL_HASH MD5=51af98ad0b7ec6fb7aa1cf96ea65fff8
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${CXXOPTS_INSTALL_PREFIX}
    LOG_DOWNLOAD ON
)

set(TARGET_PREFIX "predict")
ExternalProject_Add(
    predict
    PREFIX ${TARGET_PREFIX}
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.."
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_PREFIX}"
    INSTALL_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
        -Dxgboost_DIR:PATH=${XGBOOST_INSTALL_PREFIX}/lib/cmake/xgboost/
        -Drabit_DIR:PATH=${XGBOOST_INSTALL_PREFIX}/lib/cmake/rabit/
        -Ddmlc_DIR:PATH=${XGBOOST_INSTALL_PREFIX}/lib/cmake/dmlc/
        -Dcxxopts_DIR:PATH=${CXXOPTS_INSTALL_PREFIX}/lib/cmake/cxxopts/
    DEPENDS
        xgboost
        cxxopts
)
