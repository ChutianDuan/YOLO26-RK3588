#!/usr/bin/python3
"""Manually started real-camera stream soak. No services or unrelated PIDs touched."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("seconds must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    children, files = [], []
    env = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        env.pop(key, None)
    interrupted = False

    def signal_stop(*_):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_stop)

    def start(name, command):
        log = (root/(name+".log")).open("wb"); files.append(log)
        p = subprocess.Popen(list(map(str,command)),stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env)
        children.append(p)
        return p

    def stop(p, sig=signal.SIGTERM):
        if p.poll() is None:
            p.send_signal(sig)
            try:
                p.wait(timeout=7)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=3)

    url = "rtsp://127.0.0.1:8554/camera1"
    started = time.monotonic()
    previous = {}

    def process_sample(pid):
        base = Path(f"/proc/{pid}")
        try:
            stat = (base/"stat").read_text().rsplit(") ",1)[1].split()
            cpu = (int(stat[11])+int(stat[12]))/os.sysconf("SC_CLK_TCK")
            rss = int((base/"statm").read_text().split()[1])*os.sysconf("SC_PAGESIZE")/1048576
            now = time.monotonic()
            old = previous.get(pid)
            previous[pid] = (now,cpu)
            return dict(pid=pid,rss_mb=rss,fd_count=len(list((base/"fd").iterdir())),
                        cpu_percent=(cpu-old[1])*100/(now-old[0]) if old else None)
        except (OSError,ValueError):
            return dict(pid=pid,unavailable=True)

    def child_pids(parent):
        found = []
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                if int(stat.read_text().rsplit(") ",1)[1].split()[1]) == parent:
                    found.append(int(stat.parent.name))
            except (OSError,ValueError,IndexError):
                pass
        return found

    report = {}
    try:
        server = start("mediamtx", [project/"tools/mediamtx/v1.21.2/mediamtx",project/"yolo_run/deploy/mediamtx.yml"])
        pub = start("publisher", ["/usr/bin/python3",project/"yolo_run/deploy/publish_rtsp.py"])
        camera = start("camera", [project/"yolo_run/build-rk3588/camera_live","--preview","off","--stream","on",
                                  "--output",root/"camera.jsonl"])
        deadline = time.monotonic()+40
        while time.monotonic() < deadline:
            probe = subprocess.run(["/usr/bin/ffprobe","-v","error","-rtsp_transport","tcp","-analyzeduration","300000",
                                    "-probesize","65536","-show_streams","-of","json",url],capture_output=True,timeout=8)
            if probe.returncode == 0 and json.loads(probe.stdout).get("streams"):
                (root/"ffprobe.json").write_bytes(probe.stdout)
                break
            assert camera.poll() is None, "camera exited during startup"
            time.sleep(.25)
        else:
            raise RuntimeError("stream not available")
        reader = start("reader", ["/usr/bin/ffmpeg","-nostdin","-v","warning","-rtsp_transport","tcp","-i",url,
                                  "-map","0:v:0","-c:v","copy","-f","null","-"])
        report["startup_s"] = time.monotonic()-started
        started = time.monotonic()  # Measure the complete requested duration after publication is ready.
        last_notice = -60
        with (root/"resources.jsonl").open("w") as samples:
            while not interrupted and time.monotonic()-started < args.seconds:
                elapsed = time.monotonic()-started
                assert camera.poll() is None and server.poll() is None and pub.poll() is None and reader.poll() is None, "soak process exited"
                row = dict(elapsed_s=elapsed,camera=process_sample(camera.pid),mediamtx=process_sample(server.pid),
                           publisher=process_sample(pub.pid),reader=process_sample(reader.pid))
                publisher_children = child_pids(pub.pid)
                assert len(publisher_children) <= 1, "unbounded publisher children"
                row["ffmpeg"] = [process_sample(p) for p in publisher_children]
                try:
                    row["npu_load"] = Path("/sys/kernel/debug/rknpu/load").read_text().strip()
                except OSError as exc:
                    row["npu_load_error"] = str(exc)
                row["temperature_c"] = max(float(p.read_text())/1000 for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"))
                samples.write(json.dumps(row)+"\n"); samples.flush()
                if elapsed-last_notice >= 60:
                    print(json.dumps(dict(elapsed_s=round(elapsed,1),camera=row["camera"],temperature_c=row["temperature_c"])),flush=True)
                    last_notice = elapsed
                time.sleep(min(2,max(0,args.seconds-(time.monotonic()-started))))
        report["wall_duration_s"] = time.monotonic()-started
        stop(camera,signal.SIGINT)
        summary = None
        with (root/"camera.jsonl").open() as source:
            for line in source:
                row=json.loads(line)
                if row["type"] == "summary":
                    summary=row
        assert camera.returncode == 0 and summary and summary["status"] == "interrupted", summary
        assert summary["stream"]["status"] == "stopped" and summary["stream"]["encoded"] > 0, summary
        data = [json.loads(line) for line in (root/"resources.jsonl").read_text().splitlines()]
        stable = [r for r in data if r["elapsed_s"] >= 30] or data
        report.update(status="interrupted" if interrupted else "completed",requested_duration_s=args.seconds,summary=summary,
            stable_camera_fd_min=min(r["camera"]["fd_count"] for r in stable),
            stable_camera_fd_max=max(r["camera"]["fd_count"] for r in stable),
            stable_camera_rss_first_mb=stable[0]["camera"]["rss_mb"],stable_camera_rss_last_mb=stable[-1]["camera"]["rss_mb"],
            max_temperature_c=max(r["temperature_c"] for r in data),
            mac_rtsp="not verified on a real Mac",browser_playback="not verified in a real browser")
        (root/"report.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
    finally:
        for child in reversed(children):
            stop(child)
        for file in files:
            file.close()


if __name__ == "__main__":
    main()
