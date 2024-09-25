# %% Imports, constants and helpers

import json
from pathlib import Path
import sqlite3
# from typing import List

# import matplotlib.pylab as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(
    "/home/marvin/workspace/guut-evaluation/03_cosmic_ray_v0.3/string_utils_9fcb4a43/loops"
)
BASELINE_RESULTS_DIR = Path(
    "/home/marvin/workspace/guut-evaluation/03_cosmic_ray_v0.3/string_utils_365285bb_baseline/loops"
)
BASELINE2_RESULTS_DIR = Path(
    "/home/marvin/workspace/guut-evaluation/03_cosmic_ray_v0.3/string_utils_3fd541b_baseline2/loops"
)
SESSION_FILE = Path(
    "/home/marvin/workspace/guut-evaluation/03_cosmic_ray_v0.3/string_utils_9fcb4a43/mutants.sqlite"
)

pd.set_option("display.width", 120)

# %% Load json result files

results_json = []
for root, dirs, files in [
    *RESULTS_DIR.walk(),
    *BASELINE_RESULTS_DIR.walk(),
    *BASELINE2_RESULTS_DIR.walk(),
]:
    result_paths = (root / name for name in files if name == "result.json")
    for path in result_paths:
        result = json.loads(path.read_text())
        id = str(path.parent.name)
        if id.startswith("baseline2"):
            result["implementation"] = "baseline2"
        elif id.startswith("baseline"):
            result["implementation"] = "baseline"
        else:
            result["implementation"] = "loop"
        results_json.append(result)

# Sort!
results_json.sort(key=lambda result: result["timestamp"])

data = pd.json_normalize(results_json)

# %% Load mutants

con = sqlite3.connect(SESSION_FILE)
cur = con.cursor()
cur.execute(
    "select module_path, operator_name, occurrence, start_pos_row, end_pos_row from mutation_specs;"
)
mutants = cur.fetchall()

# %% Load mutant line info into the table

mutant_lines = {(m[0], m[1], m[2]): (m[3], m[4]) for m in mutants}
data["mutant_lines"] = data.apply(
    lambda row: mutant_lines[
        (
            row["problem.target_path"],
            row["problem.mutant_op"],
            row["problem.occurrence"],
        )
    ],
    axis=1,
)
data["mutant_lines.start"] = data["mutant_lines"].map(lambda t: t[0])
data["mutant_lines.end"] = data["mutant_lines"].map(lambda t: t[1])


# %% Add coverage info to the table


def find_killing_test(tests):
    killing_tests = [test for test in tests if test["kills_mutant"]]
    return killing_tests[0] if killing_tests else None


def get_coverage_from_test(test):
    if test is None:
        return [], []
    test_result = test.get("result")
    if test_result is None:
        return [], []
    coverage = test_result["correct"].get("coverage")
    if coverage is None:
        return [], []
    return coverage["covered_lines"], coverage["missing_lines"]


data["coverage.covered_lines"] = data["tests"].map(
    lambda tests: get_coverage_from_test(find_killing_test(tests))[0]
)
data["coverage.covered_lines_num"] = data["coverage.covered_lines"].map(len)
data["coverage.missing_lines"] = data["tests"].map(
    lambda tests: get_coverage_from_test(find_killing_test(tests))[1]
)
data["coverage.missing_lines_num"] = data["coverage.missing_lines"].map(len)


# %%

x = data[data["implementation"] == "loop"]

num_mutants = len(mutants)
total_kills = x["killed_mutants"].map(len).sum()
direct_kills = len(x[x["mutant_killed"]])
indirect_kills = total_kills - direct_kills

print(f"Total mutants: {num_mutants}")
print(f"Total mutant kills: {total_kills}")
print(f"Direct mutant kills: {direct_kills}")
print(f"Indirect mutant kills: {indirect_kills}")
print(f"Alive mutants: {num_mutants - total_kills}")

# %%


x = data[(data["implementation"] == "baseline2") & data["mutant_killed"]]
# x[x["coverage.covered_lines"].map(bool) | x["coverage.missing_lines"].map(bool)]


y = x["coverage.covered_lines_num"] / (
    x["coverage.covered_lines_num"] + x["coverage.missing_lines_num"]
)

np.reshape(y, (len(y),)).mean()

# %% Find mutants which the loop didn't kill but baseline did


def mutant_id(row):
    return (
        row["problem.target_path"],
        row["problem.mutant_op"],
        row["problem.occurrence"],
    )


loop_data = data[(data["implementation"] == "loop")]
baseline_data = data[(data["implementation"] == "baseline")]

loop_failed_mutants = set(
    loop_data[~loop_data["mutant_killed"]].apply(mutant_id, axis=1)
)
baseline_successful_mutants = set(
    baseline_data[baseline_data["mutant_killed"]].apply(mutant_id, axis=1)
)

print(len(loop_failed_mutants))
print(len(baseline_successful_mutants))

target_mutants = loop_failed_mutants.intersection(baseline_successful_mutants)

x = data[data["implementation"] != "baseline2"]
x[x.apply(mutant_id, axis=1).map(lambda id: id in target_mutants)][
    [
        "implementation",
        "problem.target_path",
        "problem.mutant_op",
        "problem.occurrence",
        "mutant_killed",
        "id",
    ]
].sort_values(["problem.target_path", "problem.mutant_op", "problem.occurrence"])
