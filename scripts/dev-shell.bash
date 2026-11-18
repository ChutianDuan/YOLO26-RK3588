# Read only by the child Bash started through scripts/dev.sh.
if [[ ${YOLO_DEV_ROLE:-} == vision ]]; then
    source /opt/miniforge3/etc/profile.d/conda.sh
    conda activate /opt/miniforge3/envs/rk3588-infer || exit "$?"
fi
cd -- "$YOLO_DEV_ROOT" || exit "$?"
PS1="(yolo:${YOLO_DEV_ROLE}) \u@\h:\w\$ "
printf 'YOLO 项目 shell：%s；输入 exit 返回原终端。\n' "$YOLO_DEV_ROLE"
