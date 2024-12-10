#!/usr/bin/env bash

# Strict bash
set -euo pipefail
IFS=$'\n\t'

# Parameters
if (( ${#} < 3 )); then
    >&2 echo "Usage: $(basename "${0}") SOCKET_NAME PATH LINENUM"
    exit 1
fi

socket_name="$1"
file_path="$(realpath "$2")"
line_num="${3:-1}"

NVIM_SOCKET="/tmp/${socket_name}.socket"
NEOVIDE_PATH="$(which neovide)"

if [ ! -S $NVIM_SOCKET ]; then
    VIM_TITLE="$socket_name" daemonize "$NEOVIDE_PATH" --no-tabs -- --listen $NVIM_SOCKET "${file_path}" "+${line_num}"
else
    nvim --server $NVIM_SOCKET --remote $file_path
    nvim --server $NVIM_SOCKET --remote-send "<ESC>${line_num}gg"
fi
