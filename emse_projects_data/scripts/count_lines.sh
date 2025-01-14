#!/usr/bin/env bash

if (( $# != 1 )); then
    >&2 echo "Usage: $(basename "${0}") EMSE_PROJECT_PATH"
    exit 1
fi

project_path="$(realpath "$1")"
project_name="$(basename "$project_path")"
src_path="$(jq -r ".\"${project_name}\"" "$(dirname "${0}")/src_paths.json")"

cd "${project_path}/${src_path}"
echo "${project_name}: $(fd -e py | wc -l)"
