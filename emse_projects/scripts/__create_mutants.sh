#!/usr/bin/env bash

for dir in *; do
    if [[ -d "${dir}" && ! ${dir} == __* ]]; then
        src_path="$(jq -r ".\"${dir}\"" __src_paths.json)"
        if [[ "${src_path}" != "null" ]]; then
            echo "Creating mutants for ${dir}"
            db_path="$(realpath "${dir}.sqlite")"
            ( cd "${dir}/${src_path}"; cosmic-ray init <(printf "[cosmic-ray]\nmodule-path=[\".\"]") "${db_path}" )
        else
            tput setaf 1; echo "Couldn't find source dir for ${dir}"; tput sgr0
        fi
    fi
done
