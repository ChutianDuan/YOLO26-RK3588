#!/bin/sh
# Project-local official binary; never edits system packages or service files.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
destination="$project_dir/tools/mediamtx/v1.21.2"
archive=mediamtx_v1.21.2_linux_arm64.tar.gz
digest=180a235374a548d9133cdd5e20f5dbd8b01317be1c86e0d7e66df585dc42ed6f
case "$(uname -m)" in aarch64|arm64) ;; *) echo 'requires native ARM64 Linux' >&2; exit 1;; esac
staging=$(mktemp -d)
trap 'rm -rf "$staging"' EXIT HUP INT TERM
curl -fsSL --retry 2 --max-time 120 -o "$staging/$archive" "https://github.com/bluenviron/mediamtx/releases/download/v1.21.2/$archive"
curl -fsSL --retry 2 --max-time 30 -o "$staging/checksums.sha256" https://github.com/bluenviron/mediamtx/releases/download/v1.21.2/checksums.sha256
cd "$staging"
printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
awk -v name="$archive" '$2 == name {print; found=1} END {if (!found) exit 1}' checksums.sha256 | sha256sum -c -
mkdir -p "$destination"
tar -xzf "$archive" -C "$destination" mediamtx LICENSE mediamtx.yml
cp checksums.sha256 "$destination/checksums.sha256"
"$destination/mediamtx" --version
