#!/usr/bin/env bash

if (( $# != 1 )); then
    >&2 echo "Usage: $(basename "${0}") EMSE_PROJECT_PATH"
    exit 1
fi

project_path="$(realpath "$1")"
project_name="$(basename "$project_path")"
src_path="$(jq -r ".\"${project_name}\"" "$(dirname "${0}")/src_paths.json")"

python_loc() {
    cloc --json "$1" | jq 'if has("Python") then .Python.code else 0 end'
}

cd "${project_path}/${src_path}"

num_files=0
num_nonempty_files=0
total_loc=0

while read f; do
    num_files=$(( num_files + 1 ))
    loc="$(python_loc "$f")"
    if [ $loc -gt 0 ]; then
        num_nonempty_files=$(( num_nonempty_files + 1 ))
        total_loc=$(( total_loc + loc ))
    fi
done < <(fd -e py)

# echo "${project_name}: $num_files"
echo "${project_name} non-empty files: $num_nonempty_files"
echo "${project_name} LoC: $total_loc"
