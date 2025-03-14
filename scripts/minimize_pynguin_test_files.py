import json
import re
import sys
from pathlib import Path

with open(sys.argv[1]) as f:
    minimized_tests = json.load(f)


TEST_CASE_REGEX = re.compile(r"def (test_case_\d+)")


for test_path in Path(".").glob("*/*/tests/*.py"):
    index = int(test_path.parent.parent.name)
    project = test_path.parent.parent.parent.name

    try:
        test_names = minimized_tests[project][str(index)]
    except KeyError:
        test_names = []
    test_names = [name.split("::") for name in test_names]
    test_names = [name[2] for name in test_names if name[1] == test_path.stem]

    text = test_path.read_text()
    blocks = text.split("\n\n")

    filtered_blocks = []
    found_test = False
    for block in blocks:
        match = re.search(TEST_CASE_REGEX, block)
        if match is None:
            filtered_blocks.append(block)
        elif match.group(1) in test_names:
            filtered_blocks.append(block)
            found_test = True

    if found_test:
        test_path.write_text("\n\n".join(filtered_blocks))
    else:
        test_path.write_text("")
