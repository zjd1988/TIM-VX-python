cmake_minimum_required(VERSION 3.4.1)
project(pytimvx)

option(ENABLE_DEBUG_LOG "enable pytimvx print debug log" ON)
if(ENABLE_DEBUG_LOG)
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=1)
endif()

if(TIMVX_INSTALL_DIR)
    set(TIM_VX_INCLUDE_PATH ${TIMVX_INSTALL_DIR}/include)
    set(TIM_VX_LIB_PATH ${TIMVX_INSTALL_DIR}/lib)
else()
    set(TIM_VX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../host_build/install/include)
    set(TIM_VX_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../host_build/install/lib)
endif()
set(PYBIND11_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/pybind11-2.9.2/include)
set(SPDLOG_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/spdlog-1.12.0/include)
set(NLOHMANN_JSON_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/nlohmann-json-3.11.2/include)
set(PYBIND11_JSON_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/pybind11_json/include)
file(GLOB_RECURSE PYTIMVX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

find_package(Python REQUIRED COMPONENTS Development)
set(CYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
set(CYTHON_LIBS ${Python_LIBRARIES})
set(PROJECT_LINK_LIBS ${PROJECT_LINK_LIBS} ${CYTHON_LIBS})
message("use system python include: ${CYTHON_INCLUDE_DIRS}")
message("use system python libs: ${CYTHON_LIBS}")

link_directories(${TIM_VX_LIB_PATH})
add_library(pytimvx SHARED ${PYTIMVX_SRC})

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-stack-protector -std=gnu99 -O2 -flax-vector-conversions -fvisibility=hidden")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-stack-protector -std=c++11 -O2 -fvisibility=hidden -fvisibility-inlines-hidden")

target_include_directories(pytimvx PRIVATE 
    ${CYTHON_INCLUDE_DIRS}
    ${TIM_VX_INCLUDE_PATH}
    ${PYBIND11_INCLUDE_PATH}
    ${SPDLOG_INCLUDE_PATH}
    ${NLOHMANN_JSON_INCLUDE_PATH}
    ${PYBIND11_JSON_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(pytimvx PRIVATE 
    tim-vx
    ${CYTHON_LIBS}
)

list(REMOVE_ITEM PYTIMVX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/pytimvx.cpp)
add_library(timvx_engine SHARED ${PYTIMVX_SRC})
target_include_directories(timvx_engine PRIVATE 
    ${TIM_VX_INCLUDE_PATH}
    ${SPDLOG_INCLUDE_PATH}
    ${NLOHMANN_JSON_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(timvx_engine PRIVATE tim-vx)
