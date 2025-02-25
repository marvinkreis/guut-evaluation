#!/usr/bin/env bash

scc --by-file -f json --include-ext py | jq '.[] | select(.Name == "Python") | .Files | map({(.Location): .Lines}) | add'
