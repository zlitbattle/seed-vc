#!/usr/bin/env python3
import argparse
import http.client
import itertools
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


logger = logging.getLogger("round_robin_proxy")


class RoundRobinProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_cycle = None

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def _proxy(self):
        upstream = next(self.upstream_cycle)
        body_len = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(body_len) if body_len > 0 else None
        parsed = urlsplit(upstream)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=600)
        path = self.path
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "connection", "keep-alive", "proxy-connection"}
        }
        headers["Host"] = f"{parsed.hostname}:{parsed.port}"
        try:
            conn.request(self.command, path, body=body, headers=headers)
            response = conn.getresponse()
            response_body = response.read()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in {"connection", "keep-alive", "transfer-encoding"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except Exception as exc:
            logger.exception("proxy_failed upstream=%s path=%s", upstream, path)
            payload = f"upstream proxy failed: {type(exc).__name__}: {exc}".encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        finally:
            conn.close()

    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)


def main():
    parser = argparse.ArgumentParser(description="Tiny round-robin HTTP reverse proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--upstream", action="append", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    RoundRobinProxy.upstream_cycle = itertools.cycle(args.upstream)
    server = ThreadingHTTPServer((args.host, args.port), RoundRobinProxy)
    logger.info("proxy_started host=%s port=%s upstreams=%s", args.host, args.port, ",".join(args.upstream))
    server.serve_forever()


if __name__ == "__main__":
    main()
