#!/usr/bin/env bash

if (( $# != 2 )); then
    >&2 echo "Usage: $(basename "${0}") EMSE_PROJECT_PATH OUT_PATH"
    exit 1
fi

project_path="$(realpath "$1")"
project_name="$(basename "$project_path")"
out_path="$(realpath "$2")"

echo "Extracting requirements.txt for $project_name"
cd "$project_path"

# Apply workarounds
export SETUPTOOLS_SCM_PRETEND_VERSION=1.0
export PBR_VERSION=1.0
if [[ "$project_name" == "isort" ]]; then
    sed -i '/deprecated_finder/d' pyproject.toml
fi

shopt -s nullglob
uv pip compile --python-version 3.8.19 requirements.txt* setup.cfg* setup.py* pyproject.toml* > "${out_path}"

# Note: some requirements need to be edited. See 'dependency_adjustment.txt'
