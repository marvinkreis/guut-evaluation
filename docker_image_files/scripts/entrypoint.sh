#!/usr/bin/env bash

# Strict bash
set -euo pipefail
IFS=$'\n\t'

if (( $# < 1 )); then
    >&2 echo "Usage: $(basename "${0}") SCRIPT_NAME"
    echo "Available scripts:"
    for script in "$(dirname ${0})"/*; do
        name="$(basename "${script}")"
        if [[ "$name" != "entrypoint.sh" ]]; then
            echo "  ${name%.*}"
        fi
    done
    exit 1
fi

script="$1"
shift
"$(dirname ${0})/${script}.sh" "$@"
