#!/usr/bin/env bash

OUT_DIR='.'

for problem in $(guut list quixbugs); do
    guut run -y --outdir "$OUT_DIR" quixbugs "$problem"
    guut run -y --baseline --outdir "$OUT_DIR" quixbugs "$problem"
done
