#!/usr/bin/env bash

for x in *; do
   jq "{$x: .}" $x/cosmic-ray-full/failing_tests.json
done | jq -s 'add'
