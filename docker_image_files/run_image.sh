#!/usr/bin/env bash

set -u

OPENAI_API_KEY=""
OPENAI_ORGANIZATION=""

mkdir /tmp/guut

docker container run --rm -it \
    --mount=type=tmpfs,target=/tmp \
    --mount=type=bind,src=/tmp/guut,target=/tmp/guut \
    -v guut-emse-results:/results \
    -v guut-emse-logs:/logs \
    -v pynguin-emse-results:/pynguin_results \
    --entrypoint "/scripts/entrypoint.sh" \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    -e OPENAI_ORGANIZATION="${OPENAI_ORGANIZATION}" \
    guut-emse "$@"

