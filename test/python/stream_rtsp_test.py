#!/usr/bin/python3
"""Serial RK3588 integration: real camera, MPP, publisher, RTSP and WHEP signaling."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urljoin

binary, fault, mediamtx, publisher, config, artifacts = map(Path, sys.argv[1:7])
app_config = Path(sys.argv[7]) if len(sys.argv) == 8 else None
app_args = ["--config", str(app_config)] if app_config else []
root = artifacts / ("integration_" + str(time.time_ns()))
root.mkdir(parents=True)
children, logs = [], []
env = dict(os.environ)
for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
    env.pop(key, None)


def port(kind):
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


rtsp, http, ice, udp = port(socket.SOCK_STREAM), port(socket.SOCK_STREAM), port(socket.SOCK_DGRAM), port(socket.SOCK_DGRAM)
url = f"rtsp://127.0.0.1:{rtsp}/camera1"
cfg = root/"mediamtx.yml"
cfg.write_text(config.read_text().replace(":8554", f":{rtsp}").replace(":8889", f":{http}").replace(":8189", f":{ice}"))


def start(name, args, **kwargs):
    log = (root/(name+".log")).open("ab")
    logs.append(log)
    child = subprocess.Popen(list(map(str,args)), stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, **kwargs)
    children.append(child)
    return child


def stop(child, sig=signal.SIGTERM):
    if child.poll() is None:
        child.send_signal(sig)
        try:
            child.wait(timeout=6)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait(timeout=3)


def rows(path):
    if not path.exists():
        return []
    # The writer may currently be writing the last line.
    lines = path.read_bytes().split(b"\n")[:-1]
    return [json.loads(line) for line in lines]


def count():
    return sum(r["type"] == "frame" for r in rows(root/"camera.jsonl"))


def child_pids(parent):
    # This kernel does not expose /proc/PID/task/PID/children.
    found = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            if int(stat.read_text().rsplit(") ",1)[1].split()[1]) == parent:
                found.append(int(stat.parent.name))
        except (OSError, ValueError, IndexError):
            pass
    return sorted(found)


def ready(timeout=35):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        assert camera.poll() is None, "camera stopped unexpectedly"
        try:
            probe = subprocess.run(["/usr/bin/ffprobe", "-v", "error", "-rtsp_transport", "tcp",
                "-analyzeduration", "300000", "-probesize", "65536", "-show_entries",
                "stream=codec_name,width,height,has_b_frames,profile", "-of", "json", url], capture_output=True, timeout=5)
            if probe.returncode == 0:
                streams = json.loads(probe.stdout).get("streams", [])
                if streams and streams[0].get("width") == 800:
                    return streams[0]
        except subprocess.TimeoutExpired:
            pass
        time.sleep(.2)
    raise AssertionError("RTSP publication did not recover; see " + str(root))


report = {}
try:
    server = start("mediamtx", [mediamtx, cfg])
    camera = start("camera", [binary, *app_args, "--preview", "off", "--stream", "on", "--stream-port", udp,
                               "--output", root/"camera.jsonl"])
    deadline = time.monotonic()+10
    while count() < 15 and time.monotonic() < deadline:
        assert camera.poll() is None
        time.sleep(.1)
    assert count() >= 15, "camera did not run without publisher"
    pubargs = ["/usr/bin/python3", publisher, "--input-port", udp, "--url", url]
    pub = start("publisher", pubargs)
    report["rtsp_stream"] = ready()
    decoded = subprocess.run(["/usr/bin/ffmpeg", "-nostdin", "-v", "error", "-rtsp_transport", "tcp",
                              "-i", url, "-frames:v", "3", "-f", "null", "-"], capture_output=True, timeout=12)
    assert decoded.returncode == 0, decoded.stderr.decode()
    report["rtsp_decode"] = "passed"
    assert urllib.request.urlopen(f"http://127.0.0.1:{http}/camera1", timeout=3).status == 200
    offer = "\r\n".join(["v=0", "o=- 1 1 IN IP4 127.0.0.1", "s=-", "t=0 0", "a=group:BUNDLE 0",
        "m=video 9 UDP/TLS/RTP/SAVPF 96", "c=IN IP4 0.0.0.0", "a=mid:0", "a=recvonly", "a=rtcp-mux",
        "a=ice-ufrag:test1234", "a=ice-pwd:testpassword012345678901", "a=ice-options:trickle",
        "a=fingerprint:sha-256 " + ":".join(["00"]*32), "a=setup:actpass", "a=rtpmap:96 H264/90000",
        "a=fmtp:96 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f", ""])
    endpoint = f"http://127.0.0.1:{http}/camera1/whep"
    response = urllib.request.urlopen(urllib.request.Request(endpoint, data=offer.encode(), headers={"Content-Type":"application/sdp"}), timeout=5)
    answer = response.read().decode()
    assert response.status == 201 and "H264/90000" in answer
    (root/"whep_answer.sdp").write_text(answer)
    urllib.request.urlopen(urllib.request.Request(urljoin(endpoint,response.headers["Location"]), method="DELETE"),timeout=3).close()
    report["whep_signaling"] = "passed; media playback not asserted"
    # Stop draining a real client's stdout; its slow delivery must not back up inference.
    slow = subprocess.Popen(["/usr/bin/ffmpeg", "-nostdin", "-v", "error", "-rtsp_transport", "tcp", "-i", url,
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    children.append(slow)
    before = count(); time.sleep(1); assert count() >= before+15
    stop(slow); slow.stdout.close(); report["slow_client"] = "inference/output continued"
    stop(server)
    before = count(); time.sleep(1); assert camera.poll() is None and count() >= before+15
    server = start("mediamtx_restart", [mediamtx,cfg])
    ready(); report["mediamtx_restart"] = "recovered without restarting camera"
    pids = child_pids(pub.pid); assert len(pids) == 1, pids
    os.kill(pids[0],signal.SIGKILL)
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        current = child_pids(pub.pid)
        assert len(current) <= 1
        if current and current != pids:
            break
        time.sleep(.1)
    else:
        raise AssertionError("FFmpeg child did not restart")
    ready(); report["ffmpeg_restart"] = "recovered with one child"
    stop(pub)
    before = count(); time.sleep(1); assert count() >= before+15
    pub = start("publisher_restart",pubargs)
    ready(); report["publisher_restart"] = "recovered without restarting camera"
    stop(camera,signal.SIGINT)
    summary = rows(root/"camera.jsonl")[-1]
    if app_config:
        metadata = rows(root/"camera.jsonl")[0]
        assert (metadata["model_width"], metadata["model_height"], metadata["num_classes"]) == (640,640,80), metadata
        assert Path(metadata["model"]).name == "yolo26s_coco_fp16.rknn", metadata
        assert metadata["padding"]["top"] == metadata["padding"]["bottom"] == 80, metadata
        report["model_metadata"] = metadata
    assert summary["status"] == "interrupted" and summary["stream"]["status"] == "stopped", summary
    assert summary["inference_count"] > 0 and summary["stream"]["encoded"] > 0
    report["summary"] = summary
    stop(pub); stop(server)
    for name, extra in [("slow_encoder", ["--test-encode-delay-ms","100"]),
                        ("failed_encoder", ["--test-fail-encode-after","3"])]:
        path = root/(name+".jsonl")
        p = start(name,[fault,*app_args,"--preview","off","--stream","on","--stream-port",udp,
                       "--max-frames","60","--output",path,*extra])
        assert p.wait(timeout=15) == 0
        result = rows(path)[-1]
        assert result["frame_count"] == 60 and result["inference_count"] > 0, result
        if name == "slow_encoder":
            assert result["stream"]["pending_dropped"] > 0 and result["stream"]["status"] == "stopped"
        else:
            assert result["stream"]["status"] == "error" and "injected" in result["stream"]["error"]
        report[name] = result["stream"]
    for args in [["--stream","on","--decoder","opencv"], ["--stream-port","0"], ["--stream-port","65536"],
                 ["--stream-bitrate","1"], ["--stream-bitrate","14000001"], ["--stream-bitrate","999999999999"], ["--stream","yes"]]:
        p = subprocess.run([str(binary),*app_args,"--preview","off",*args],capture_output=True,env=env,timeout=5)
        assert p.returncode != 0
    report["cli_invalid_options"] = "passed"
    (root/"report.json").write_text(json.dumps(report,indent=2))
    print("PASS RTSP decode, WHEP signaling, process recovery, slow client/encoder, error isolation and CLI:",root)
finally:
    for child in reversed(children):
        stop(child)
    for log in logs:
        log.close()
