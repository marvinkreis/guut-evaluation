import json
import re
import sys
from pathlib import Path

# def is_pynguin_run_excluded(project, index):
#     index = int(index)
#     if project == "pdir2":  # Breaks Pynguin, because pdir2 replaces its own module with a function.
#         return True
#     if (project, index) in [
#         ("flutils", 6),  # Tests delete the package under test.
#         ("flutils", 9),  # Tests delete the package under test.
#         ("flutils", 10),  # Breaks coverage.py.
#         ("flutils", 20),  # Breaks coverage.py.
#         ("flake8", 4),  # Tests cause pytest to raise an OSError. Most mutants are killed even though the tests pass.
#         ("apimd", 24),  # Missing in the results.
#     ]:
#         return True
#     return False

with open(sys.argv[1]) as f:
    minimized_tests = json.load(f)


TEST_CASE_REGEX = re.compile(r"def (test_case_\d+)")


for test_path in Path(".").glob("*/*/tests/*.py"):
    index = int(test_path.parent.parent.name)
    project = test_path.parent.parent.parent.name

    # if is_pynguin_run_excluded(project, index):
    #     continue
    #
    # if project not in minimized_tests:
    #     continue
    #
    # if str(index) not in minimized_tests[project]:
    #     continue
    #
    # if is_pynguin_run_excluded(project, index):
    #     continue
    #
    # test_names = minimized_tests[project][str(index)]

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
