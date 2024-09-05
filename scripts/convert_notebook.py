#!/usr/bin/env python

import json
import sys
import uuid
from pathlib import Path

if len(sys.argv) < 2:
    print("Convert from zed's repl format to jupyter's notebook format.")
    print(f"Usage: {sys.argv[0]} script.py")
    sys.exit(1)

in_path = Path(sys.argv[1])
out_name = in_path.name.replace(".py", ".ipynb")
assert out_name != in_path.name
out_path = Path(".") / out_name

print(f"Converting {in_path} to {out_path}.")

code = in_path.read_text()

sections = []
current_name = ""
current_lines = []
for line in code.splitlines():
    if line.startswith("# %%"):
        if current_lines:
            sections.append((current_name, "\n".join(current_lines).strip()))
            current_lines = []
        current_name = line[4:].strip()
    current_lines.append(line)
if current_lines:
    sections.append((current_name, "\n".join(current_lines).strip()))

cells = []
for section in sections:
    if section[0]:
        cells.append(
            {
                "cell_type": "markdown",
                "id": str(uuid.uuid4()),
                "metadata": {},
                "outputs": [],
                "source": f"#### {section[0]}",
            }
        )
    if section[1]:
        cells.append(
            {
                "cell_type": "code",
                "id": str(uuid.uuid4()),
                "metadata": {},
                "outputs": [],
                "source": section[1],
            }
        )

json_obj = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}

with out_path.open("w") as file:
    json.dump(json_obj, file)
