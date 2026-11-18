#!/usr/bin/python3
"""Supervise one independent, stream-copy FFmpeg publisher; never own YOLO/MTX."""
import argparse
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit


def log(event, **fields):
    print(json.dumps(dict(event=event, monotonic_s=time.monotonic(), **fields)), flush=True)


def command(port, url):
    return ["/usr/bin/ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-progress", "pipe:1", "-stats_period", "1", "-fflags", "nobuffer",
            "-analyzeduration", "1000000", "-probesize", "262144", "-i",
            f"udp://127.0.0.1:{port}?localaddr=127.0.0.1&buffer_size=65536"
            "&fifo_size=512&overrun_nonfatal=0&timeout=5000000",
            "-map", "0:v:0", "-an", "-c:v", "copy", "-muxdelay", "0",
            "-rtsp_transport", "tcp", "-pkt_size", "1200", "-f", "rtsp", url]


def terminate(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=2)


def supervise(argv, stopped, stall_timeout=10):
    backoff = 1
    while not stopped.is_set():
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=None, start_new_session=True, bufsize=0)
        log("started", pid=child.pid)
        started = last_advance = time.monotonic()
        last_out = None
        pending = b""
        reason = "exit"
        try:
            with selectors.DefaultSelector() as selector:
                os.set_blocking(child.stdout.fileno(), False)
                selector.register(child.stdout, selectors.EVENT_READ)
                while child.poll() is None and not stopped.is_set():
                    for key, _ in selector.select(timeout=.25):
                        chunk = os.read(key.fileobj.fileno(), 4096)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        pending += chunk
                        if len(pending) > 65536:
                            raise RuntimeError("oversized FFmpeg progress line")
                        while b"\n" in pending:
                            raw, pending = pending.split(b"\n", 1)
                            name, _, value = raw.decode("utf-8", "replace").partition("=")
                            if name == "out_time_us":
                                try:
                                    out = int(value)
                                except ValueError:
                                    continue
                                if last_out is not None and out > last_out:
                                    last_advance = time.monotonic()
                                last_out = out
                            elif name == "progress":
                                log("progress", pid=child.pid, out_time_us=last_out)
                    if time.monotonic() - last_advance >= stall_timeout:
                        reason = "stalled"
                        break
        finally:
            terminate(child)
            child.stdout.close()
        log("stopped", pid=child.pid, returncode=child.returncode, reason=reason)
        if stopped.is_set():
            break
        if last_advance - started >= 30:
            backoff = 1
        log("retry", delay_s=backoff)
        stopped.wait(backoff)
        backoff = min(backoff * 2, 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-port", type=int, default=12340)
    parser.add_argument("--url", default="rtsp://127.0.0.1:8554/camera1")
    args = parser.parse_args()
    url = urlsplit(args.url)
    if not 1 <= args.input_port <= 65535 or url.scheme != "rtsp" or not url.hostname or url.path in ("", "/"):
        parser.error("require UDP port 1..65535 and an rtsp://host/path URL")
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    supervise(command(args.input_port, args.url), stopped)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"publisher: {exc}", file=sys.stderr)
        sys.exit(1)
