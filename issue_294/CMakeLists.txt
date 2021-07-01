cmake_minimum_required(VERSION 3.9.2)
project(testing_issues)

add_executable(rnn_wings.exe src/rnn_wings.cpp)

find_package(EDDL REQUIRED)
target_link_libraries(rnn_wings.exe PUBLIC EDDL::eddl)
