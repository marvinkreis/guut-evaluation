#!/usr/bin/env bash

if (( $# != 2 )); then
    >&2 echo "Usage: $(basename "${0}") EMSE_PROJECT_NAME OUT_PATH"
    exit 1
fi

project_name="$1"
venv_path="$2"
requirements_path="$(dirname "${0}")/../requirements/${project_name}.txt"

echo "Creating venv for $project_name"
uv venv "$venv_path" --python cpython-3.10.14-linux-x86_64-gnu
source "${venv_path}/bin/activate"
uv pip install -r "$requirements_path" coverage pytest pytest-timeout

# pyMonet might not work with python 3.10
