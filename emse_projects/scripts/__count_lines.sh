#!/usr/bin/env bash

for dir in *; do
    if [[ -d "${dir}" && ! ${dir} == __* ]]; then
        src_path="$(jq -r ".\"${dir}\"" __src_paths.json)"
        if [[ "${src_path}" != "null" ]]; then
            if [[ "$(dirname "${src_path}")" == "src" ]]; then
                src_path="$(basename "${src_path}")"
                ( cd "${dir}/src"; fd -e py -x wc -l)
            else
                ( cd "${dir}"; fd -e py -x wc -l)
            fi
        else
            tput setaf 1; echo "Couldn't find source dir for ${dir}"; tput sgr0
        fi
    fi
done
