#!/usr/bin/env bash

for dir in *; do
    if [[ -d "${dir}" && ! ${dir} == __* ]]; then
        echo "Creating venv for ${dir}"
        ( cd "${dir}"; sh "../__venv_scripts/${dir}.sh" )
    fi
done
