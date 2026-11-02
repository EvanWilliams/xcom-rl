cmake_minimum_required(VERSION 3.5)
project(bridge)

add_library(bridge STATIC SocketServer.cpp)

target_include_directories(bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (WIN32)
    # Winsock2 provides the AF_UNIX socket implementation on Windows.
    target_link_libraries(bridge PUBLIC Ws2_32)
else()
    # POSIX uses native BSD sockets; std::thread needs the pthread runtime.
    find_package(Threads REQUIRED)
    target_link_libraries(bridge PUBLIC Threads::Threads)
endif()
