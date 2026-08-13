cmake_minimum_required(VERSION 3.8)
project(touch_teleop)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)


add_executable(touch_teleop_node src/touch_teleop_node.cpp)
add_executable(touch_twist src/touch_twist_node.cpp)

# add_executable(servo_control_node src/servo_control_node.cpp)

target_include_directories(touch_teleop_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(touch_teleop_node
  rclcpp
  geometry_msgs
  sensor_msgs
)

# target_include_directories(servo_control_node PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>
# )


# ament_target_dependencies(servo_control_node
#   rclcpp
#   sensor_msgs
#   fairino_hardware
#   fairino_msgs
# )

ament_target_dependencies(touch_twist
  rclcpp
  geometry_msgs
  Eigen3
)

target_link_libraries(touch_twist HD HDU Eigen3::Eigen rt)

# OpenHaptics libraries + Eigen3
target_link_libraries(touch_teleop_node HD HDU Eigen3::Eigen)

install(TARGETS
  touch_teleop_node
  touch_twist

  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
