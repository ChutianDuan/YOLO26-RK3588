#!/usr/bin/python3
import contextlib
import importlib.util
import io
import json
import os
import sys
import threading

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("publisher", sys.argv[1])
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
cmd = publisher.command(12340, "rtsp://127.0.0.1:8554/camera1")
assert cmd[cmd.index("-c:v")+1] == "copy" and "-re" not in cmd
assert "fifo_size=512" in cmd[cmd.index("-i")+1]

stopped = threading.Event()
timer = threading.Timer(.8, stopped.set)
output = io.StringIO()
timer.start()
try:
    with contextlib.redirect_stdout(output):
        publisher.supervise(["/usr/bin/python3", "-c",
                              "import time; print('out_time_us=1',flush=True); time.sleep(20)"],
                             stopped, stall_timeout=.2)
finally:
    stopped.set()
    timer.join()
rows = [json.loads(line) for line in output.getvalue().splitlines()]
assert any(r["event"] == "stopped" and r["reason"] == "stalled" for r in rows), rows
assert any(r["event"] == "retry" and r["delay_s"] == 1 for r in rows), rows
for row in rows:
    if row["event"] == "started":
        try:
            os.kill(row["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("publisher leaked a child")
print("PASS stream-copy arguments, stalled-child watchdog, bounded retry and child cleanup")
