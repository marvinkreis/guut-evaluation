#!/usr/bin/env bash

for dir in *; do
    if [[ -d "${dir}" && ! ${dir} == __* ]]; then
        if [ -d "${dir}/.venv" ]; then
            echo "found venv for ${dir}"
        else
            tput setaf 1; echo -n "missing"; tput sgr0; echo " venv for ${dir}"
        fi
    fi
done
