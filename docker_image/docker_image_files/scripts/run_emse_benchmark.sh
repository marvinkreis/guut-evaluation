#!/usr/bin/env bash

# Strict bash
set -euo pipefail
IFS=$'\n\t'

if (( $# != 2 )); then
    >&2 echo "Usage: $(basename "${0}") PRESET_NAME EMSE_PROJECT_NAME"
    exit 1
fi

preset_name=$1
project_name=$2

source "/guut/.venv/bin/activate"
src_path="$(python -c "import json; paths=json.load(open('/emse_projects/scripts/src_paths.json')); print(paths['$project_name'])")"
session_file="/emse_projects/mutants_sampled/${project_name}.sqlite"
full_module_path="/emse_projects/code/${project_name}/${src_path}"

cd "/guut"
python -m guut run -y --preset "$preset_name" --python-interpreter "/emse_projects/venvs/$project_name/bin/python" cosmic-ray-individual-mutants "$session_file" "$full_module_path"
