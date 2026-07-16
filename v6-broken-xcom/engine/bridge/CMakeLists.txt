cmake_minimum_required(VERSION 3.5)
project(bridge)

add_library(bridge STATIC SocketServer.cpp)

target_include_directories(bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bridge PUBLIC Ws2_32)
