#!/usr/bin/env bash

if (( $# != 2 )); then
    >&2 echo "Usage: $(basename "${0}") EMSE_PROJECT_PATH OUT_PATH"
    exit 1
fi

project_path="$(realpath "$1")"
project_name="$(basename "$project_path")"
out_path="$(realpath "$2")"
src_path="$(jq -r ".\"${project_name}\"" "$(dirname "${0}")/src_paths.json")"

echo "Creating mutants for ${project_name}"
cd "${project_path}/${src_path}"
cosmic-ray init <(printf "[cosmic-ray]\nmodule-path=[\".\"]") "${out_path}"
