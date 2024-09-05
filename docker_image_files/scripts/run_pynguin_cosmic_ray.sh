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
session_file="/emse_projects/mutants_sampled/${project_name}.sqlite"
full_module_path="/emse_projects/code/${project_name}/${src_path}"

if [[ ! -d "${result_dir}/cosmic-ray" ]]; then
    mkdir "${result_dir}/cosmic-ray"
fi

cd "${full_module_path}/.."
source "/emse_projects/venvs/${project_name}/bin/activate"
python /cosmic-ray/run_pytest_tests.py baseline "${result_dir}/cosmic-ray/failing_tests.json" "${result_dir}/tests/"*.py

cat << EOF > /cosmic-ray/test_command.sh
#!/usr/bin/env bash

cd "${full_module_path}/.."
source "/emse_projects/venvs/${project_name}/bin/activate"
python /cosmic-ray/run_pytest_tests.py test "${result_dir}/cosmic-ray/failing_tests.json" "${result_dir}/tests/"*.py
EOF
chmod +x /cosmic-ray/test_command.sh

if [[ ! -f "${result_dir}/cosmic-ray/mutants.sqlite" ]]; then
    cp "${session_file}" "${result_dir}/cosmic-ray/mutants.sqlite"
fi

cd "${full_module_path}"
source "/guut/.venv/bin/activate"
cosmic-ray --verbosity INFO exec /cosmic-ray/session.toml "${result_dir}/cosmic-ray/mutants.sqlite"
