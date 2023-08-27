cmake_minimum_required(VERSION 3.18)

project(robust_plane_fit)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(${PROJECT_NAME}
    INTERFACE
    ./src/iteratively_reweighted_least_squares.hpp
    ./src/m_estimator.hpp
)

target_link_libraries(${PROJECT_NAME}
    INTERFACE
    Eigen3::Eigen
)

set(EXECUTABLE_NAME main)

add_executable(${EXECUTABLE_NAME} ./src/main.cpp)

target_link_libraries(${EXECUTABLE_NAME}
    PRIVATE
    ${PROJECT_NAME}
)