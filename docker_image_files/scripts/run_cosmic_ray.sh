#!/usr/bin/env bash

# Strict bash
set -euo pipefail
IFS=$'\n\t'

if (( $# < 2 )) || (( $# > 3)); then
    >&2 echo "Usage: $(basename "${0}") RESULT_ID EMSE_PROJECT_NAME [--continue] [-x]"
    exit 1
fi

result_dir="/results/$1"
project_name=$2

continue=0
stop_after_first_fail=""
while (( $# > 2)); do
    if [[ "$3" == "--continue" ]]; then
        continue=1
        shift
    fi
    if [[ "$3" == "-x" ]]; then
        stop_after_first_fail="-x"
        shift
    fi
done

source "/guut/.venv/bin/activate"
src_path="$(python -c "import json; paths=json.load(open('/emse_projects/scripts/src_paths.json')); print(paths['$project_name'])")"
session_file="/emse_projects/mutants_sampled/${project_name}.sqlite"
full_module_path="/emse_projects/code/${project_name}/${src_path}"

if [[ "$stop_after_first_fail" == "-x" ]]; then
    result_cosmic_ray_dir="${result_dir}/cosmic-ray"
else
    result_cosmic_ray_dir="${result_dir}/cosmic-ray-full"
fi

if [[ ! -d "$result_cosmic_ray_dir" ]]; then
    mkdir "$result_cosmic_ray_dir"
fi
if (( $continue == 0 )) && [[ -f "$result_cosmic_ray_dir/failing_tests.json" ]]; then
    rm "$result_cosmic_ray_dir/failing_tests.json"
fi

if (( $continue == 0 )); then
    cd "${full_module_path}/.."
    source "/emse_projects/venvs/${project_name}/bin/activate"
    python /cosmic-ray/run_tests.py baseline "$result_cosmic_ray_dir/failing_tests.json" "${result_dir}/tests/"*.py
fi

cat << EOF > /cosmic-ray/test_command.sh
#!/usr/bin/env bash

cd "${full_module_path}/.."
source "/emse_projects/venvs/${project_name}/bin/activate"
python /cosmic-ray/run_tests.py test "$result_cosmic_ray_dir/failing_tests.json" $stop_after_first_fail "${result_dir}/tests/"*.py
EOF
chmod +x /cosmic-ray/test_command.sh

if (( $continue == 0 )); then
    cp "${session_file}" "$result_cosmic_ray_dir/mutants.sqlite"
fi

cd "${full_module_path}"
source "/guut/.venv/bin/activate"
cosmic-ray --verbosity INFO exec /cosmic-ray/session.toml "$result_cosmic_ray_dir/mutants.sqlite"
