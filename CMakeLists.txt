cmake_minimum_required(VERSION 3.25)
project(Gauss_Legendre)

set(CMAKE_CXX_STANDARD 17)

# Prerequisites

find_package(Eigen3 REQUIRED)

link_directories(${Eigen_INCLUDE_DIRS})

add_executable(Gauss_Legendre gauss_legendre.cpp)

#target_link_libraries(Gauss_Legendre ${Eigen_LIBRARIES})
target_link_libraries(Gauss_Legendre Eigen3::Eigen)