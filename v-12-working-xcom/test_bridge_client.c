#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <afunix.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    char defaultPath[256];
    const char *tmp = getenv("TEMP");
    if (!tmp) tmp = getenv("TMP");
    if (!tmp) tmp = "C:\\Windows\\Temp";
    snprintf(defaultPath, sizeof(defaultPath), "%s\\openxcom_bridge.sock", tmp);
    const char *path = argc > 1 ? argv[1] : defaultPath;

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }

    SOCKET s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) {
        fprintf(stderr, "socket() failed: %d\n", WSAGetLastError());
        WSACleanup();
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    printf("Connecting to: %s\n", path);
    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "connect() failed: %d\n", WSAGetLastError());
        closesocket(s);
        WSACleanup();
        return 1;
    }
    printf("Connected! Waiting for messages (max 20)...\n");
    fflush(stdout);

    char buf[4096];
    int count = 0;
    while (count < 20) {
        int n = recv(s, buf, sizeof(buf) - 1, 0);
        if (n <= 0) { printf("Connection closed.\n"); break; }
        buf[n] = '\0';
        printf("%s", buf);
        fflush(stdout);
        count++;
    }

    closesocket(s);
    WSACleanup();
    return 0;
}
