#!/usr/bin/python3
"""Validate actual MPP H264 output with the system FFmpeg decoder and ffprobe."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(args):
    return subprocess.run(args, check=True, capture_output=True, timeout=40).stdout


with tempfile.TemporaryDirectory(prefix="yolo-stream-media-") as name:
    root = Path(name)
    run([sys.argv[1], name])
    probe = json.loads(run(["/usr/bin/ffprobe", "-v", "error", "-show_streams",
                            "-show_frames", "-of", "json", str(root/"encoded.ts")]))
    stream = probe["streams"][0]
    assert (stream["width"], stream["height"], stream["has_b_frames"], stream["level"]) == (800, 600, 0, 31), stream
    assert "Baseline" in stream["profile"] and stream["color_range"] == "tv", stream
    assert stream["color_space"] == "smpte170m", stream
    expected = json.loads((root/"expected.json").read_text())
    frames = probe["frames"]
    assert len(frames) == len(expected) == 64, (len(frames), len(expected))
    for f, e in zip(frames, expected):
        assert int(f["pts"]) == e["pts90"], (f["pts"], e)
        assert f["pict_type"] in ("I", "P") and bool(f["key_frame"]) == e["keyframe"]
    keys = [i for i, e in enumerate(expected) if e["keyframe"]]
    assert keys[0] == 0 and len(keys) >= 3, keys
    assert all(expected[b]["pts90"]-expected[a]["pts90"] <= 93000 for a, b in zip(keys, keys[1:])), keys
    reference = run(["/usr/bin/ffmpeg", "-v", "error", "-i", str(root/"reference.png"),
                     "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    decoded = run(["/usr/bin/ffmpeg", "-v", "error", "-i", str(root/"encoded.ts"),
                   "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    assert len(decoded) == len(reference) == 800*600*3
    errors = []
    for bar in range(5):
        per_channel = [0, 0, 0]
        count = 0
        for y in range(350, 450):
            for x in range(bar*160+20, bar*160+140):
                p = (y*800+x)*3
                for ch in range(3):
                    per_channel[ch] += abs(decoded[p+ch]-reference[p+ch])
                count += 1
        errors.append([v/count for v in per_channel])
    assert max(v for bar in errors for v in bar) <= 8, errors
    # The known green box and text must survive at their original coordinates.
    def green_count(image, x0, y0, w, h):
        return sum(image[(y*800+x)*3+1] > 120 and
                   image[(y*800+x)*3+1] > image[(y*800+x)*3+2]+40
                   for y in range(y0,y0+h) for x in range(x0,x0+w))
    assert green_count(decoded,338,158,116,136) > 500
    assert green_count(decoded,339,137,85,17) > 50
    print(json.dumps({"frames":len(frames), "keyframe_indices":keys, "color_mae_bgr":errors,
                      "timestamps":"preserved including gaps", "known_box_and_id":"passed"}))
