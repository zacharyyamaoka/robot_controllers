# robot_controllers C++ implementations (CONTRACT amendment 2).
#
#   robot_controllers::headers   INTERFACE: include/robot_controllers/*.hpp, header-only laws for C++ users
#                                (the bam_ws moteus controller node links this into its control thread)
#   robot_controllers_jtc        SHARED: the C ABI the Python adapter loads with ctypes
#
# The Python adapter does not need this file: it compiles jtc_capi.cpp itself on first use (see
# controllers/joint_trajectory_cpp.py). This is the standard build for other C++ projects.
cmake_minimum_required(VERSION 3.16)
project(robot_controllers_cpp LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(robot_controllers_headers INTERFACE)
add_library(robot_controllers::headers ALIAS robot_controllers_headers)
target_include_directories(robot_controllers_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(robot_controllers_headers INTERFACE cxx_std_17)
# WHY no FMA contraction: the laws use (1 - w) * a + w * b exactly as the Python reference; a fused
# multiply-add rounds differently and the parity tests (<= 1e-12, bit-exact today) would fail.
target_compile_options(robot_controllers_headers INTERFACE -ffp-contract=off)

add_library(robot_controllers_jtc SHARED jtc_capi.cpp)
target_link_libraries(robot_controllers_jtc PRIVATE robot_controllers::headers)
target_compile_options(robot_controllers_jtc PRIVATE -Wall -Wextra -Wpedantic)
