#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <afunix.h>
#include <io.h>
#include <stdio.h>
#include <string.h>

/* Standalone AF_UNIX echo server for diagnosing bridge connectivity */
int main(int argc, char *argv[]) {
    const char *path = argc > 1 ? argv[1] :
        "C:\\Users\\evan\\Documents\\Claude Code Stuff\\xcom-reinforcement-learning"
        "\\engine\\openxcom\\build\\bin\\openxcom_bridge.sock";

    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);

    _unlink(path);

    SOCKET srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv == INVALID_SOCKET) { fprintf(stderr, "socket: %d\n", WSAGetLastError()); return 1; }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    printf("binding to: %s\n", addr.sun_path);
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "bind: %d\n", WSAGetLastError()); return 1;
    }
    if (listen(srv, 5) == SOCKET_ERROR) {
        fprintf(stderr, "listen: %d\n", WSAGetLastError()); return 1;
    }
    printf("listening...\n"); fflush(stdout);

    SOCKET cli = accept(srv, NULL, NULL);
    if (cli == INVALID_SOCKET) { fprintf(stderr, "accept: %d\n", WSAGetLastError()); return 1; }
    printf("client connected\n"); fflush(stdout);

    const char *msg = "{\"funds\":1234567}\n";
    send(cli, msg, (int)strlen(msg), 0);
    printf("sent test message\n");

    closesocket(cli);
    closesocket(srv);
    _unlink(path);
    WSACleanup();
    return 0;
}
