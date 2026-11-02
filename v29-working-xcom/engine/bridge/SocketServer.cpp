// Cross-platform AF_UNIX socket server for the RL bridge.
// Windows uses Winsock2 + afunix.h; POSIX (Linux/Debian, macOS) uses the
// native BSD sockets API.  All platform differences are confined to the
// small shim block below so the server logic stays single-sourced.
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <winsock2.h>
    #include <afunix.h>
    #include <io.h>
    #include <stdlib.h>
#else
    #include <sys/socket.h>
    #include <sys/select.h>
    #include <sys/un.h>
    #include <unistd.h>
    #include <cstring>
    #include <cerrno>
#endif
#include <algorithm>
#include <cstdio>
#include "SocketServer.h"

// ── POSIX shim: map the handful of Winsock names used below onto BSD sockets ──
#ifndef _WIN32
using SOCKET = int;
using TIMEVAL = struct timeval;
static const int INVALID_SOCKET = -1;
static const int SOCKET_ERROR    = -1;
#ifndef UNIX_PATH_MAX
    #define UNIX_PATH_MAX 108
#endif
// send(): suppress SIGPIPE so a disconnected client can't kill the process.
#define BRIDGE_SEND_FLAGS MSG_NOSIGNAL
static inline int  closesocket(int s)   { return ::close(s); }
static inline int  WSAGetLastError()    { return errno; }
static inline int  _unlink(const char* p) { return ::unlink(p); }
static inline void Sleep(unsigned ms)   { ::usleep(ms * 1000u); }
// strcpy_s(dst[N], src) equivalent, deducing the destination array size.
template<size_t N> static inline void strcpy_s(char (&dst)[N], const char* src) {
    std::strncpy(dst, src, N - 1);
    dst[N - 1] = '\0';
}
#else
    #define BRIDGE_SEND_FLAGS 0
#endif

#define BRIDGE_LOG(fmt, ...) do { \
    FILE *_f = fopen("bridge_debug.txt", "a"); \
    if (_f) { fprintf(_f, fmt "\n", ##__VA_ARGS__); fclose(_f); } \
} while(0)

namespace Bridge {

SocketServer& SocketServer::instance() {
    static SocketServer inst;
    return inst;
}

SocketServer::~SocketServer() {
    stop();
}

void SocketServer::start(const std::string& socketPath) {
    BRIDGE_LOG("start() called, path=%s", socketPath.c_str());
    if (_running.load()) { BRIDGE_LOG("already running"); return; }

    // Build absolute socket path; must fit in sockaddr_un.sun_path (108 bytes).
    // If the caller passes a bare filename, place it in the temp dir to
    // guarantee a short path.
    std::string resolvedPath = socketPath;
    if (socketPath.find(':') == std::string::npos && socketPath.find('\\') == std::string::npos &&
        socketPath.find('/') == std::string::npos) {
#ifdef _WIN32
        const char *tmp = getenv("TEMP");
        if (!tmp) tmp = getenv("TMP");
        if (!tmp) tmp = "C:\\Windows\\Temp";
        resolvedPath = std::string(tmp) + "\\" + socketPath;
#else
        const char *tmp = getenv("TMPDIR");
        if (!tmp) tmp = "/tmp";
        resolvedPath = std::string(tmp) + "/" + socketPath;
#endif
    }
    if (resolvedPath.size() >= UNIX_PATH_MAX) {
        BRIDGE_LOG("socket path too long (%zu >= %d): %s",
                   resolvedPath.size(), UNIX_PATH_MAX, resolvedPath.c_str());
        return;
    }
    _socketPath = resolvedPath;
    BRIDGE_LOG("resolved path=%s", _socketPath.c_str());

#ifdef _WIN32
    WSADATA wsa;
    int wsaErr = WSAStartup(MAKEWORD(2, 2), &wsa);
    BRIDGE_LOG("WSAStartup: %d", wsaErr);
    if (wsaErr != 0) return;
#endif

    SOCKET sock = socket(AF_UNIX, SOCK_STREAM, 0);
    BRIDGE_LOG("socket(): %s (err=%d)", sock == INVALID_SOCKET ? "INVALID" : "ok", WSAGetLastError());
    if (sock == INVALID_SOCKET) {
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    _unlink(_socketPath.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strcpy_s(addr.sun_path, _socketPath.c_str());
    BRIDGE_LOG("sun_path set to: %s", addr.sun_path);

    int bindRes = bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    BRIDGE_LOG("bind(): %d (err=%d)", bindRes, WSAGetLastError());
    if (bindRes == SOCKET_ERROR) {
        closesocket(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    int listenRes = listen(sock, SOMAXCONN);
    BRIDGE_LOG("listen(): %d (err=%d)", listenRes, WSAGetLastError());
    if (listenRes == SOCKET_ERROR) {
        closesocket(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return;
    }

    _listenSock = static_cast<uintptr_t>(sock);
    _running.store(true);
    _acceptThread = std::thread(&SocketServer::acceptLoop, this);
    BRIDGE_LOG("bridge started ok");
}

void SocketServer::stop() {
    if (!_running.exchange(false)) return;

    SOCKET sock = static_cast<SOCKET>(_listenSock);
    _listenSock = ~uintptr_t(0);
    closesocket(sock);

    if (_acceptThread.joinable())
        _acceptThread.join();

    {
        std::lock_guard<std::mutex> lock(_clientsMutex);
        for (auto &cc : _clients)
            closesocket(static_cast<SOCKET>(cc.sock));
        _clients.clear();
    }

    _unlink(_socketPath.c_str());
#ifdef _WIN32
    WSACleanup();
#endif
}

void SocketServer::acceptLoop() {
    while (_running.load()) {
        SOCKET client = accept(static_cast<SOCKET>(_listenSock), nullptr, nullptr);
        if (client == INVALID_SOCKET) break;

        {
            std::lock_guard<std::mutex> lock(_clientsMutex);
            ClientConn cc;
            cc.sock = static_cast<uintptr_t>(client);
            _clients.push_back(cc);
        }
        _newClientCount.fetch_add(1);
    }
}

bool SocketServer::takeNewClient() {
    return _newClientCount.exchange(0) > 0;
}

void SocketServer::broadcast(const std::string& msg) {
    std::lock_guard<std::mutex> lock(_clientsMutex);

    std::vector<size_t> deadIdx;
    for (size_t i = 0; i < _clients.size(); ++i) {
        SOCKET s = static_cast<SOCKET>(_clients[i].sock);
        // Non-blocking guard: skip if the kernel send buffer has no room.
        // In headless mode Python reads immediately so select() always returns
        // writable (no drops).  In visible mode this prevents the SDL loop from
        // stalling when the Python consumer lags behind.
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(s, &wfds);
        TIMEVAL tv = {0, 0};
        // Windows ignores select()'s first arg; POSIX needs (maxfd + 1).
#ifdef _WIN32
        int selRes = select(0, nullptr, &wfds, nullptr, &tv);
#else
        int selRes = select(static_cast<int>(s) + 1, nullptr, &wfds, nullptr, &tv);
#endif
        if (selRes <= 0)
            continue;  // buffer full — drop this broadcast
        if (send(s, msg.c_str(), static_cast<int>(msg.size()), BRIDGE_SEND_FLAGS) == SOCKET_ERROR) {
            closesocket(s);
            deadIdx.push_back(i);
        }
    }

    for (size_t ri = deadIdx.size(); ri-- > 0; )
        _clients.erase(_clients.begin() + deadIdx[ri]);
}

std::vector<std::string> SocketServer::pollCommands() {
    std::vector<std::string> result;
    std::lock_guard<std::mutex> lock(_clientsMutex);
    if (_clients.empty()) return result;

    fd_set rfds;
    FD_ZERO(&rfds);
    int nfds = 0;
    for (auto &cc : _clients) {
        SOCKET s = static_cast<SOCKET>(cc.sock);
        FD_SET(s, &rfds);
        if (static_cast<int>(s) + 1 > nfds) nfds = static_cast<int>(s) + 1;
    }

    TIMEVAL tv = {0, 0};
#ifdef _WIN32
    int selRes = select(0, &rfds, nullptr, nullptr, &tv);
#else
    int selRes = select(nfds, &rfds, nullptr, nullptr, &tv);
#endif
    if (selRes <= 0)
        return result;

    std::vector<size_t> deadIdx;
    for (size_t i = 0; i < _clients.size(); ++i) {
        SOCKET s = static_cast<SOCKET>(_clients[i].sock);
        if (!FD_ISSET(s, &rfds)) continue;

        char tmp[4096];
        int n = recv(s, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            closesocket(s);
            deadIdx.push_back(i);
            continue;
        }
        _clients[i].rbuf.append(tmp, static_cast<size_t>(n));

        size_t pos;
        while ((pos = _clients[i].rbuf.find('\n')) != std::string::npos) {
            std::string line = _clients[i].rbuf.substr(0, pos);
            _clients[i].rbuf.erase(0, pos + 1);
            if (!line.empty()) result.push_back(line);
        }
    }

    for (size_t ri = deadIdx.size(); ri-- > 0; )
        _clients.erase(_clients.begin() + deadIdx[ri]);

    return result;
}

// Blocks up to timeout_ms milliseconds waiting for a command from any client.
// Returns immediately when a command arrives.  The lock is held only briefly —
// we release it before calling select() so broadcast() and acceptLoop() can
// proceed during the wait.
std::vector<std::string> SocketServer::pollCommandsBlocking(int timeout_ms) {
    // Snapshot the connected sockets under the lock (brief hold).
    fd_set rfds;
    FD_ZERO(&rfds);
    int nfds = 0;
    {
        std::lock_guard<std::mutex> lock(_clientsMutex);
        if (_clients.empty()) {
            // No Python client yet — sleep briefly so the game loop doesn't spin.
            Sleep(timeout_ms < 10 ? timeout_ms : 10);
            return {};
        }
        for (auto &cc : _clients) {
            SOCKET s = static_cast<SOCKET>(cc.sock);
            FD_SET(s, &rfds);
            if (static_cast<int>(s) + 1 > nfds) nfds = static_cast<int>(s) + 1;
        }
    }
    // Wait outside the lock — broadcast() and acceptLoop() can run freely.
    TIMEVAL tv = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
#ifdef _WIN32
    select(0, &rfds, nullptr, nullptr, &tv);
#else
    select(nfds, &rfds, nullptr, nullptr, &tv);
#endif
    // Non-blocking read to drain whatever arrived.
    return pollCommands();
}

} // namespace Bridge
