#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Bridge {

class SocketServer {
public:
    static SocketServer& instance();

    void start(const std::string& socketPath);
    void stop();
    void broadcast(const std::string& msg);
    std::vector<std::string> pollCommands();
    bool takeNewClient(); // returns true (once) when a new client has connected since last call

private:
    SocketServer() = default;
    ~SocketServer();
    SocketServer(const SocketServer&) = delete;
    SocketServer& operator=(const SocketServer&) = delete;

    void acceptLoop();

    struct ClientConn {
        uintptr_t sock;
        std::string rbuf;
    };

    std::string _socketPath;
    uintptr_t _listenSock = ~uintptr_t(0); // INVALID_SOCKET sentinel
    std::thread _acceptThread;
    std::mutex _clientsMutex;
    std::vector<ClientConn> _clients;
    std::atomic<bool> _running{false};
    std::atomic<int> _newClientCount{0};
};

} // namespace Bridge
