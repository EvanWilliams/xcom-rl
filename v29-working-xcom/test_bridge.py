"""Connect to the openxcom bridge socket and print incoming funds updates."""
import socket
import os
import sys

# AF_UNIX = 1 on all platforms; Python on Windows omits the constant but Winsock supports it
AF_UNIX = getattr(socket, "AF_UNIX", 1)

SOCKET_PATH = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "openxcom_bridge.sock")

def main():
    print(f"Connecting to {SOCKET_PATH}")
    with socket.socket(AF_UNIX, socket.SOCK_STREAM) as s:
        try:
            s.connect(SOCKET_PATH)
        except FileNotFoundError:
            print("Socket not found — is the engine running?")
            sys.exit(1)
        except ConnectionRefusedError:
            print("Connection refused — bridge not listening?")
            sys.exit(1)

        print("Connected. Waiting for updates (Ctrl+C to stop)...\n")
        buf = b""
        count = 0
        while True:
            data = s.recv(4096)
            if not data:
                print("Server closed connection.")
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                msg = line.decode("utf-8").strip()
                if msg:
                    count += 1
                    print(f"[{count:>4}] {msg}")

if __name__ == "__main__":
    main()
