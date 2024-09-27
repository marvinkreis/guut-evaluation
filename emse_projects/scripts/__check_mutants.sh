#!/usr/bin/env bash

for dir in *; do
    if [[ -d "${dir}" && ! ${dir} == __* ]]; then
        if [ -f "${dir}.sqlite" ]; then
            echo "found mutants for ${dir}"
        else
            tput setaf 1; echo -n "missing"; tput sgr0; echo " mutants for ${dir}"
        fi
    fi
done
