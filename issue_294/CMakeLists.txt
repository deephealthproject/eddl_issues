cmake_minimum_required(VERSION 3.9.2)
project(testing_issues)

add_executable(rnn_wings.exe src/rnn_wings.cpp)
add_executable(fully_conn_wings.exe src/fully_conn_wings.cpp)

find_package(EDDL REQUIRED)
target_link_libraries(rnn_wings.exe PUBLIC EDDL::eddl)
target_link_libraries(fully_conn_wings.exe PUBLIC EDDL::eddl)
