#!/usr/bin/env bash

# A venv for each project is assumed to be set up under
# "guut-evaluation/emse_projects_data/venvs/$project_name".
# This is only needed for running tests.

# Strict bash
set -euo pipefail
IFS=$'\n\t'

# Parameters
if (( ${#} < 1 )); then
    >&2 echo "Usage: $(basename "${0}") row_index"
    exit 1
fi

INDEX="$1"
WORKDIR='/tmp/inspect_mutant'
REPO_PATH="$(realpath "$(dirname "${0}")/..")"
GUUT_PATH="$(realpath "$REPO_PATH/../guut")" # change as needed
TESTS_PATH="$REPO_PATH/samples/tests"

mkdir -p "${WORKDIR}"
cd "$WORKDIR"

# Read info from samples.csv
csv_row=$(rg "^${INDEX}," "$REPO_PATH/samples/sampled_mutants.csv" )
IFS=, read -r row_index project target_path mutant_op occurrence rest <<< "$csv_row"
src_path="$(jq -r ".\"${project}\"" "${REPO_PATH}/emse_projects_data/scripts/src_paths.json")"

# Set up two fresh instances of the project in the $WORKDIR
if [ -d "${project}" ]; then
    rm -r "${project}"
fi
if [ -d "${project}_mutant" ]; then
    rm -r "${project}_mutant"
fi
cp -r "${REPO_PATH}/emse_projects/${project}" "${project}"
cp -r "${REPO_PATH}/emse_projects/${project}" "${project}_mutant"

# Apply mutant to one instance of the project
(
    cd "${project}_mutant/${src_path}"
    source "${GUUT_PATH}/.venv/bin/activate"
    cosmic-ray apply "$target_path" "$mutant_op" "$occurrence"
)

# Compute mutant diff and find mutant line
set +e
git diff -U10 --no-index "${project}/${src_path}/${target_path}" "${project}_mutant/${src_path}/${target_path}" > mutant.diff
mutant_line="$(git diff -U0 --no-index "${project}/${src_path}/${target_path}" "${project}_mutant/${src_path}/${target_path}" | rg -o '@@ [-+]([0-9]+)[, ]' -r '$1')"
set -e


# Write test script and empty test file to the $WORKDIR
long_id="$(echo "${project}_${target_path}_${mutant_op}_${occurrence}" | tr -- "-/." "_")"
touch test.py

cat << EOF > run_test.sh
#!/usr/bin/env bash

long_path="\$(realpath ${long_id})"
test_path="\$(realpath test.py)"

if [[ "\$1" == "save" ]]; then
    cp test.py "\${long_path}.py"
fi

(
    cd "${project}/${src_path}/.."
    source ${REPO_PATH}/emse_projects_data/venvs/${project}/bin/activate
    if [[ "\$1" == "save" ]]; then
        echo "Running test on baseline:" | tee "\${long_path}_log_baseline.txt"
        PYTHONDONTWRITEBYTECODE=1 python -u -m pytest -s "\${test_path}" 2>&1 | tee -a "\${long_path}_log_baseline.txt"
    else
        echo "Running test on baseline:"
        PYTHONDONTWRITEBYTECODE=1 python -u -m pytest -s "\${test_path}"
    fi

    echo
    echo

    cd - > /dev/null
    cd "${project}_mutant/${src_path}/.."
    source ${REPO_PATH}/emse_projects_data/venvs/${project}/bin/activate
    if [[ "\$1" == "save" ]]; then
        echo "Running test on mutant:" | tee "\${long_path}_log_mutant.txt"
        PYTHONDONTWRITEBYTECODE=1 python -u -m pytest -s "\${test_path}" 2>&1 | tee -a "\${long_path}_log_mutant.txt"
    else
        echo "Running test on mutant:"
        PYTHONDONTWRITEBYTECODE=1 python -u -m pytest -s "\${test_path}"
    fi
)
EOF
chmod +x run_test.sh

# Open relevant files in editors
alias neovide_instance="$REPO_PATH/samples/neovide_instance.sh"
neovide_instance project "${project}/${src_path}/${target_path}" $mutant_line
neovide_instance mutant "mutant.diff" 1
neovide_instance test "test.py" 1
if [ -f "${TESTS_PATH}/${long_id}.py" ]; then
    neovide_instance test_done "${TESTS_PATH}/${long_id}.py"  1
    neovide_instance test_done_log_baseline "${TESTS_PATH}/${long_id}_log_baseline.txt"  1
    neovide_instance test_done_log_mutant "${TESTS_PATH}/${long_id}_log_mutant.txt"  1
fi
