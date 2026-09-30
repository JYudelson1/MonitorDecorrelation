"""Transparent TCP load balancer: listen on one port, spread connections over several backend ports.

Usage:
    uv run python scripts/load_balancer.py --listen-port 8002 --backend-ports 8003 8004 8005

Works at the TCP level: bytes are piped through untouched (no HTTP parsing, no header rewriting), so
any request type works, including streaming/SSE. Each incoming *connection* goes to the backend with
the fewest open connections (ties broken round-robin). An HTTP keep-alive connection therefore stays
on one backend for its lifetime; clients firing many parallel requests open many connections, which
get spread evenly.

Stdlib only (uses uvloop if installed). Raises the open-file soft limit to the hard limit so thousands
of concurrent connections (2 fds each) don't hit EMFILE.
"""

import argparse
import asyncio
import itertools
import resource
import socket
import sys
import time

BUFFER_SIZE = 256 * 1024
DOWN_COOLDOWN_S = 5.0  # after a failed connect, try a backend only if all others fail too


class Balancer:
    def __init__(self, backend_host: str, backend_ports: list[int]):
        self.backend_host = backend_host
        self.backend_ports = backend_ports
        self.active = {p: 0 for p in backend_ports}
        self.total = {p: 0 for p in backend_ports}
        self.failures = {p: 0 for p in backend_ports}
        self.down_until = {p: 0.0 for p in backend_ports}
        self._rr = itertools.cycle(range(len(backend_ports)))

    def _pick_order(self) -> list[int]:
        """Healthy backends first, then fewest active connections; round-robin start breaks ties."""
        start = next(self._rr)
        rotated = self.backend_ports[start:] + self.backend_ports[:start]
        now = time.monotonic()
        # Stable sort keeps the rotation for ties.
        return sorted(rotated, key=lambda p: (self.down_until[p] > now, self.active[p]))

    async def handle(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter):
        _set_nodelay(client_writer)
        backend = None
        # Try backends in preference order; fall through to the next one if a connect fails
        # (nothing has been read from the client yet, so retrying is safe).
        # Pending connects count as active, else a backend slow to accept looks idle and gets piled on.
        for port in self._pick_order():
            self.active[port] += 1
            try:
                backend_reader, backend_writer = await asyncio.open_connection(
                    self.backend_host, port, limit=BUFFER_SIZE
                )
            except OSError as e:
                self.active[port] -= 1
                self.failures[port] += 1
                now = time.monotonic()
                if self.down_until[port] <= now:  # log once per outage, not once per connection
                    print(f"[lb] backend {port} connect failed, deprioritising: {e}", file=sys.stderr, flush=True)
                self.down_until[port] = now + DOWN_COOLDOWN_S
                continue
            except BaseException:
                self.active[port] -= 1
                raise
            if self.down_until[port]:
                print(f"[lb] backend {port} is back", file=sys.stderr, flush=True)
                self.down_until[port] = 0.0
            backend = port
            break
        if backend is None:
            client_writer.close()
            return

        _set_nodelay(backend_writer)
        self.total[backend] += 1
        try:
            await asyncio.gather(
                _pipe(client_reader, backend_writer),
                _pipe(backend_reader, client_writer),
            )
        finally:
            self.active[backend] -= 1
            for w in (backend_writer, client_writer):
                w.close()
            for w in (backend_writer, client_writer):
                try:
                    await w.wait_closed()
                except (OSError, ConnectionError):
                    pass

    async def report(self, interval: float):
        while True:
            await asyncio.sleep(interval)
            parts = [
                f"{p}: active={self.active[p]} total={self.total[p]} connect_fail={self.failures[p]}"
                for p in self.backend_ports
            ]
            print("[lb] " + " | ".join(parts), flush=True)


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Copy bytes until EOF, then half-close the write side so the peer sees EOF too."""
    try:
        while True:
            data = await reader.read(BUFFER_SIZE)
            if not data:
                break
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (OSError, ConnectionError, RuntimeError):
        # Peer reset / already closed: tear down both directions.
        writer.close()


def _set_nodelay(writer: asyncio.StreamWriter):
    sock = writer.get_extra_info("socket")
    if sock is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass


def _raise_fd_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


async def main(args):
    balancer = Balancer(args.backend_host, args.backend_ports)
    server = await asyncio.start_server(
        balancer.handle,
        host=args.listen_host,
        port=args.listen_port,
        backlog=args.backlog,
        limit=BUFFER_SIZE,
        reuse_address=True,
    )
    print(
        f"[lb] listening on {args.listen_host}:{args.listen_port} -> "
        f"{args.backend_host}:{args.backend_ports} (fd limit {_raise_fd_limit()})",
        flush=True,
    )
    if args.report_every > 0:
        asyncio.create_task(balancer.report(args.report_every))
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listen-port", type=int, required=True, help="port clients send requests to")
    parser.add_argument("--backend-ports", type=int, nargs="+", required=True, help="ports to route to")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backlog", type=int, default=16384, help="listen backlog (capped by net.core.somaxconn)")
    parser.add_argument("--report-every", type=float, default=60.0, help="seconds between stats lines; 0 = off")
    args = parser.parse_args()
    if args.listen_port in args.backend_ports:
        parser.error("--listen-port must not be one of --backend-ports")

    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
