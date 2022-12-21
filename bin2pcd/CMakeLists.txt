cmake_minimum_required(VERSION 3.1)
project(bin2pcd)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")

find_package(PCL 1.10 REQUIRED)
find_package(Eigen3)
find_package(Boost COMPONENTS program_options REQUIRED )

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
)
 
# 加入Boost setting
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(bin2pcd bin2pcd.cpp)
add_executable(merge_pc merge_pc.cpp)
add_executable(filterpcd filterpcd.cpp)
add_executable(visualizedense visualizedense.cpp)

target_link_libraries (bin2pcd ${PCL_LIBRARIES} /usr/local/lib/libyaml-cpp.a)
target_link_libraries (merge_pc ${PCL_LIBRARIES} /usr/local/lib/libyaml-cpp.a)
target_link_libraries (filterpcd ${PCL_LIBRARIES} /usr/local/lib/libyaml-cpp.a)
target_link_libraries (visualizedense ${PCL_LIBRARIES} /usr/local/lib/libyaml-cpp.a)
