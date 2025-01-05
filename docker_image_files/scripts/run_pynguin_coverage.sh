#!/usr/bin/env bash

# Strict bash
set -euo pipefail
IFS=$'\n\t'

if (( $# != 2 )); then
    >&2 echo "Usage: $(basename "${0}") EMSE_PROJECT_NAME RESULT_NUM"
    exit 1
fi

project_name=$1
result_dir="/pynguin_results/${project_name}/${2}"

source "/guut/.venv/bin/activate"
src_path="$(python -c "import json; paths=json.load(open('/emse_projects/scripts/src_paths.json')); print(paths['$project_name'])")"
full_module_path="/emse_projects/code/${project_name}/${src_path}"

cd "${full_module_path}"
module_name=$(basename "$PWD")
cd ..

cat << EOF > .coveragerc
[run]
concurrency = multiprocessing
branch = True
include = $(basename "${full_module_path}")/*
EOF

source "/emse_projects/pynguin_venvs/${project_name}/bin/activate"
python /cosmic-ray/run_pytest_tests.py test --cov="${module_name}" "${result_dir}/cosmic-ray/failing_tests.json" "${result_dir}/tests/"*.py

if [[ ! -d "${result_dir}/coverage" ]]; then
    mkdir "${result_dir}/coverage"
fi

coverage combine || true
coverage json
mv coverage.json "${result_dir}/coverage/coverage.json"
coverage report -i > "${result_dir}/coverage/coverage.txt"
rm .coverage
