cmake_minimum_required(VERSION 3.4.1)
project(timvx)

set(TIM_VX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../host_build/install/include)
set(TIM_VX_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../host_build/install/lib)
set(PYBIND11_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/pybind11-2.9.2/include)
set(SPDLOG_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/spdlog-1.12.0/include)
file(GLOB_RECURSE SRC  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

find_package(Python REQUIRED COMPONENTS Development)
set(CYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
set(CYTHON_LIBS ${Python_LIBRARIES})
set(PROJECT_LINK_LIBS ${PROJECT_LINK_LIBS} ${CYTHON_LIBS})
message("use system python include: ${CYTHON_INCLUDE_DIRS}")
message("use system python libs: ${CYTHON_LIBS}")

link_directories(${TIM_VX_LIB_PATH})
add_library(timvx SHARED ${SRC})

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-stack-protector -std=gnu99 -O2 -flax-vector-conversions -fvisibility=hidden")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-stack-protector -std=c++11 -O2 -fvisibility=hidden -fvisibility-inlines-hidden")

target_include_directories(timvx PUBLIC 
    ${CYTHON_INCLUDE_DIRS}
    ${TIM_VX_INCLUDE_PATH}
    ${PYBIND11_INCLUDE_PATH}
    ${SPDLOG_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(timvx PUBLIC 
    tim-vx
    ${CYTHON_LIBS}
)

