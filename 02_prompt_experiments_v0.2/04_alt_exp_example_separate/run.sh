#!/usr/bin/env bash

OUT_DIR="$(dirname $0)"

for problem in $(guut list quixbugs); do
    guut run -y --outdir "$OUT_DIR" --altexp quixbugs "$problem"
done
