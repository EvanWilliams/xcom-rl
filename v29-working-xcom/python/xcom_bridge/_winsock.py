"""AF_UNIX socket client for the RL bridge.

On Windows this talks to Winsock2 directly through ctypes: Python 3.9+ added
socket.AF_UNIX support on Windows, but it depends on internal CPython details
that vary across builds, so ctypes matches exactly what the working C test
client does and has no external dependencies.

On POSIX (Linux/Debian, macOS) the stdlib ``socket`` module supports AF_UNIX
natively, so we use it directly.  Both branches expose the same five helpers
(``open_socket``/``recv_ready``/``recv_bytes``/``send_bytes``/``close_socket``)
operating on an opaque handle, so callers are platform-agnostic.
"""
from __future__ import annotations

import sys

if sys.platform != "win32":
    # ── POSIX branch: native AF_UNIX via the stdlib socket module ────────────
    import select as _select
    import socket as _socket

    _SEND_TIMEOUT_MS = 5_000  # engine must drain its buffer within 5 s or it is treated as dead

    def open_socket(path: str) -> "_socket.socket":
        """Connect to a Unix-domain socket at *path*.  Returns the socket object."""
        encoded = path.encode("utf-8")
        if len(encoded) >= 108:  # sockaddr_un.sun_path is 108 bytes
            raise ValueError(f"Socket path too long (must be < 108 bytes): {path!r}")
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.connect(path)  # raises ConnectionRefusedError/FileNotFoundError if engine isn't up
        return s

    def recv_ready(sock: "_socket.socket", timeout_ms: int = 200) -> bool:
        """Return True if data is available on *sock* within *timeout_ms* milliseconds."""
        r, _, _ = _select.select([sock], [], [], timeout_ms / 1000.0)
        return bool(r)

    def recv_bytes(sock: "_socket.socket", bufsize: int = 8192) -> "bytes | None":
        """Receive up to *bufsize* bytes.  Returns None on EOF."""
        data = sock.recv(bufsize)
        return data if data else None

    def send_bytes(sock: "_socket.socket", data: bytes) -> None:
        """Send all of *data* over *sock*; raises OSError on failure or send timeout."""
        view = memoryview(data)
        sent = 0
        while sent < len(data):
            _, w, _ = _select.select([], [sock], [], _SEND_TIMEOUT_MS / 1000.0)
            if not w:
                raise OSError(
                    f"send() timed out after {_SEND_TIMEOUT_MS // 1000}s "
                    "— engine receive buffer full (engine may be hung)"
                )
            n = sock.send(view[sent:sent + 65536])
            if n <= 0:
                raise OSError("send() failed")
            sent += n

    def close_socket(sock: "_socket.socket") -> None:
        try:
            sock.close()
        except OSError:
            pass

else:
    # ── Windows branch: Winsock2 via ctypes ──────────────────────────────────
    import ctypes

    _ws2 = ctypes.WinDLL("ws2_32")

    class _WSADATA(ctypes.Structure):
        _fields_ = [
            ("wVersion",       ctypes.c_uint16),
            ("wHighVersion",   ctypes.c_uint16),
            ("szDescription",  ctypes.c_char * 257),
            ("szSystemStatus", ctypes.c_char * 129),
            ("iMaxSockets",    ctypes.c_uint16),
            ("iMaxUdpDg",      ctypes.c_uint16),
            ("lpVendorInfo",   ctypes.c_char_p),
        ]

    class _SockAddrUn(ctypes.Structure):
        _fields_ = [
            ("sun_family", ctypes.c_uint16),
            ("sun_path",   ctypes.c_char * 108),   # UNIX_PATH_MAX = 108
        ]

    # fd_set for select(): count + 64 socket handles (Winsock default FD_SETSIZE=64)
    class _FdSet(ctypes.Structure):
        _fields_ = [
            ("fd_count", ctypes.c_uint),
            ("fd_array", ctypes.c_size_t * 64),
        ]

    class _Timeval(ctypes.Structure):
        _fields_ = [
            ("tv_sec",  ctypes.c_long),
            ("tv_usec", ctypes.c_long),
        ]

    # Function signatures — must match ws2_32 exports exactly.
    # SOCKET is UINT_PTR on the calling platform (64-bit on 64-bit Python).
    _ws2.WSAStartup.argtypes = [ctypes.c_uint16, ctypes.POINTER(_WSADATA)]
    _ws2.WSAStartup.restype  = ctypes.c_int
    _ws2.WSACleanup.argtypes = []
    _ws2.WSACleanup.restype  = ctypes.c_int
    _ws2.WSAGetLastError.argtypes = []
    _ws2.WSAGetLastError.restype  = ctypes.c_int
    _ws2.socket.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _ws2.socket.restype  = ctypes.c_size_t
    _ws2.connect.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
    _ws2.connect.restype  = ctypes.c_int
    _ws2.recv.argtypes = [ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    _ws2.recv.restype  = ctypes.c_int
    _ws2.send.argtypes = [ctypes.c_size_t, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    _ws2.send.restype  = ctypes.c_int
    _ws2.select.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(_FdSet), ctypes.POINTER(_FdSet), ctypes.POINTER(_FdSet),
        ctypes.POINTER(_Timeval),
    ]
    _ws2.select.restype  = ctypes.c_int
    _ws2.closesocket.argtypes = [ctypes.c_size_t]
    _ws2.closesocket.restype  = ctypes.c_int

    _AF_UNIX        = 1
    _SOCK_STREAM    = 1
    _INVALID_SOCKET = ctypes.c_size_t(-1).value
    _SOCKET_ERROR   = -1

    def open_socket(path: str) -> int:
        """Connect to a Unix-domain socket at *path*.  Returns the raw socket handle."""
        wsa = _WSADATA()
        rc = _ws2.WSAStartup(0x0202, ctypes.byref(wsa))
        if rc != 0:
            raise OSError(f"WSAStartup failed: {rc}")

        sock = _ws2.socket(_AF_UNIX, _SOCK_STREAM, 0)
        if sock == _INVALID_SOCKET:
            err = _ws2.WSAGetLastError()
            _ws2.WSACleanup()
            raise OSError(f"socket() failed: Winsock error {err}")

        encoded = path.encode("utf-8")
        if len(encoded) >= 108:
            _ws2.closesocket(sock)
            _ws2.WSACleanup()
            raise ValueError(f"Socket path too long (must be < 108 bytes): {path!r}")

        addr = _SockAddrUn()
        addr.sun_family = _AF_UNIX
        addr.sun_path = encoded

        rc = _ws2.connect(sock, ctypes.byref(addr), ctypes.sizeof(addr))
        if rc == _SOCKET_ERROR:
            err = _ws2.WSAGetLastError()
            _ws2.closesocket(sock)
            _ws2.WSACleanup()
            raise ConnectionRefusedError(
                f"connect() failed: Winsock error {err}  "
                f"(10061=connection refused, 10053=connection aborted — is the engine running?)"
            )

        return sock

    _SEND_TIMEOUT_MS = 5_000   # engine must drain buffer within 5 s or it is treated as dead

    def recv_ready(sock: int, timeout_ms: int = 200) -> bool:
        """Return True if data is available on *sock* within *timeout_ms* milliseconds."""
        fds = _FdSet()
        fds.fd_count = 1
        fds.fd_array[0] = sock
        tv = _Timeval(tv_sec=timeout_ms // 1000, tv_usec=(timeout_ms % 1000) * 1000)
        result = _ws2.select(0, ctypes.byref(fds), None, None, ctypes.byref(tv))
        return result > 0

    def _write_ready(sock: int, timeout_ms: int) -> bool:
        """Return True if *sock* can accept a write within *timeout_ms* milliseconds."""
        fds = _FdSet()
        fds.fd_count = 1
        fds.fd_array[0] = sock
        tv = _Timeval(tv_sec=timeout_ms // 1000, tv_usec=(timeout_ms % 1000) * 1000)
        result = _ws2.select(0, None, ctypes.byref(fds), None, ctypes.byref(tv))
        return result > 0

    def recv_bytes(sock: int, bufsize: int = 8192) -> "bytes | None":
        """Receive up to *bufsize* bytes.  Returns None on EOF."""
        buf = ctypes.create_string_buffer(bufsize)
        n = _ws2.recv(sock, buf, bufsize - 1, 0)
        if n <= 0:
            return None
        return bytes(buf.raw[:n])

    def send_bytes(sock: int, data: bytes) -> None:
        """Send all of *data* over *sock*; raises OSError on failure or send timeout."""
        view = memoryview(data)
        sent = 0
        while sent < len(data):
            if not _write_ready(sock, _SEND_TIMEOUT_MS):
                raise OSError(
                    f"send() timed out after {_SEND_TIMEOUT_MS // 1000}s "
                    "— engine receive buffer full (engine may be hung)"
                )
            chunk = bytes(view[sent:sent + 65536])
            n = _ws2.send(sock, chunk, len(chunk), 0)
            if n == _SOCKET_ERROR:
                raise OSError(f"send() failed: Winsock error {_ws2.WSAGetLastError()}")
            sent += n

    def close_socket(sock: int) -> None:
        _ws2.closesocket(sock)
        _ws2.WSACleanup()
