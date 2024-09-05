# %% Imports, constants and helpers

from functools import partial, reduce
import json
from pathlib import Path
import sqlite3
from typing import List, Literal, Union

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os

pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

RESULTS_DIR = Path(os.getcwd()) / "loops"
_evaluation_dir = next(iter([path for path in Path(os.getcwd()).parents if path.name == "guut-evaluation"]))
SESSION_FILE = Path(_evaluation_dir) / "emse_projects/mutants_sampled/python-string-utils.sqlite"


# %% Load json result files

results_json = []
for root, dirs, files in RESULTS_DIR.walk():
    result_paths = (root / name for name in files if name == "result.json")
    for path in result_paths:
        result = json.loads(path.read_text())
        results_json.append(result)

# Sort!
results_json.sort(key=lambda result: result["timestamp"])

data = pd.json_normalize(results_json)

# %% Load mutants

con = sqlite3.connect(SESSION_FILE)
cur = con.cursor()
cur.execute("select module_path, operator_name, occurrence, start_pos_row, end_pos_row from mutation_specs;")
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

# %% Helper functions


def get_result(exp_or_test: Union["Experiment", "Test"], mutant: bool = False) -> "ExecutionResult":
    if exp_or_test is None:
        return None

    result = exp_or_test.get("result")
    if result is None:
        return None

    if not mutant:
        return result.get("test_correct") or result.get("correct")
    else:
        return result.get("test_mutant") or result.get("mutant")


def find_killing_test(tests: List["Test"]) -> "Test":
    killing_tests = [test for test in tests if test["kills_mutant"]]
    return killing_tests[0] if killing_tests else None


def subtract_coverage(x: List[int], y: List[int]) -> List[int]:
    acc = set(x)
    acc.intersection_update(y)
    return list(acc)


# %% Check whether each run covers the mutant


def covers_mutant(row, experiments_or_tests: Literal["experiments", "tests"]):
    start, end = row["mutant_lines"]
    mutant_lines = set(range(start, end + 1))
    return any(
        set(coverage["covered_lines"]).intersection(mutant_lines)
        for exp in row[experiments_or_tests]
        if (exec_result := get_result(exp)) and (coverage := exec_result["coverage"])
    )


data["experiment_covers_mutant"] = data.apply(partial(covers_mutant, experiments_or_tests="experiments"), axis=1)
data["test_covers_mutant"] = data.apply(partial(covers_mutant, experiments_or_tests="tests"), axis=1)
data["mutant_covered"] = data["experiment_covers_mutant"] | data["test_covers_mutant"]

# %% Compute coverage from killing tests


def get_coverage_from_test(test):
    exec_result = get_result(test)
    if (exec_result is None) or (exec_result["coverage"] is None):
        return [], []
    coverage = exec_result["coverage"]
    return coverage["covered_lines"], coverage["missing_lines"]


data["coverage.covered_lines"] = data["tests"].map(lambda tests: get_coverage_from_test(find_killing_test(tests))[0])
data["coverage.missing_lines"] = data["tests"].map(lambda tests: get_coverage_from_test(find_killing_test(tests))[1])

# %% Plot Coverage


def add_coverage(x: List[int], y: List[int]) -> List[int]:
    acc = set(x)
    acc.update(y)
    return list(acc)


add_coverage_np = np.frompyfunc(add_coverage, 2, 1)

for target_path in np.unique(data[["problem.target_path"]]):
    target_data = data[data["problem.target_path"] == target_path]

    all_lines = set(reduce(add_coverage, target_data["coverage.covered_lines"], []))
    all_lines.update(reduce(add_coverage, target_data["coverage.missing_lines"], []))

    if not all_lines:
        continue

    acc_coverage = add_coverage_np.accumulate(target_data["coverage.covered_lines"])
    acc_coverage_percent = acc_coverage.map(lambda c: len(c) / len(all_lines))

    x = np.arange(len(acc_coverage))
    fig, ax = plt.subplots(layout="constrained", figsize=(8, 4))
    ax.set_ylim(0, 1)
    ax.plot(x, acc_coverage_percent, 0.6, label="accumulated coverage")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(loc="upper left")
    ax.text(0, 0.8, target_path)
    plt.show()

# %% Compute Token Usage

data["usage.prompt_tokens"] = data["conversation"].map(
    lambda c: sum(msg["usage"]["prompt_tokens"] for msg in c if msg["role"] == "assistant")
)
data["usage.completion_tokens"] = data["conversation"].map(
    lambda c: sum(msg["usage"]["completion_tokens"] for msg in c if msg["role"] == "assistant")
)
data["usage.cached_tokens"] = data["conversation"].map(
    lambda c: sum(msg["usage"].get("cached_tokens", 0) for msg in c if msg["role"] == "assistant")
)
data["usage.total_tokens"] = data["usage.prompt_tokens"] + data["usage.completion_tokens"]

# Add cost in $ for gpt-4o-mini
# prompt: $0.150 / 1M input tokens
# cached prompt: $0.075 / 1M input tokens
# completion: $0.600 / 1M input tokens
data["usage.cost"] = (
    ((data["usage.prompt_tokens"] - data["usage.cached_tokens"]) * 0.150)
    + (data["usage.cached_tokens"] * 0.075)
    + (data["usage.completion_tokens"] * 0.600)
) / 1_000_000

# %% Token usage mean

data[["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]].mean()

# %% Token usage sum

data[["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]].sum()

# %% Count messages, experiments, tests

data["num_completions"] = data["conversation"].map(
    lambda conv: len([msg for msg in conv if msg["role"] == "assistant"])
)
data["num_equivalences"] = data["conversation"].map(
    lambda conv: len([msg for msg in conv if msg["tag"] == "claimed_equivalent"])
)
data["num_experiments"] = data["experiments"].map(len)
data["num_tests"] = data["tests"].map(len)
data["num_turns"] = data["num_experiments"] + data["num_tests"]  # TODO: rename this
data["num_messages"] = data["conversation"].map(lambda msgs: len([msg for msg in msgs if msg["role"] == "assistant"]))

data["num_invalid_experiments"] = data["experiments"].map(
    lambda exps: len([exp for exp in exps if not exp["validation_result"]["valid"]])
)
data["num_invalid_tests"] = data["tests"].map(
    lambda tests: len([test for test in tests if not test["validation_result"]["valid"]])
)

# %% Count message types

relevant_msg_tags = [
    "experiment_stated",
    "experiment_doesnt_compile",
    "experiment_results_given",
    "test_instructions_given",
    "test_stated",
    "test_invalid",
    "test_doesnt_detect_mutant",
    "done",
    "incomplete_response",
    "aborted",
]

for tag in relevant_msg_tags:
    data[f"tag.{tag}"] = data["conversation"].map(lambda conv: len([msg for msg in conv if msg["tag"] == tag]))


# %% Compute test LOC


def estimate_loc(test):
    if test is None:
        return None
    return len([line for line in test["code"].splitlines() if line.strip() and not line.strip().startswith("#")])


data["test_loc"] = data["tests"].map(lambda tests: estimate_loc(find_killing_test(tests)))

# %% Compute number of import errors


def count_import_errors(exps_or_tests):
    return len(
        [
            exp
            for exp in exps_or_tests
            if (exec_result := get_result(exp)) and "ModuleNotFoundError" in exec_result["output"]
        ]
    )


data["num_experiment_import_errors"] = data["experiments"].map(count_import_errors)
data["num_test_import_errors"] = data["tests"].map(count_import_errors)

# %% Count number of debugger scripts

data["num_debugger_scripts"] = data["experiments"].map(
    lambda exps: len([exp for exp in exps if exp["debugger_script"]])
)

# %% Percentage of experiments with debugger scripts

data["num_debugger_scripts"].sum() / data["num_experiments"].sum()


# %% Number of turns

fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
ax.set_xticks([1])
ax.set_xticklabels(["# Turns"])
ax.violinplot(data["num_turns"], showmeans=True)
ax.legend(["Number of Turns"], loc="upper left")
plt.show()

# %% Number of turns per successful / unsuccessful run

turns_success = data["num_turns"][data["mutant_killed"]]
turns_equivalent = data["num_turns"][data["claimed_equivalent"]]
turns_fail = data["num_turns"][(~data["mutant_killed"]) & (~data["claimed_equivalent"])]

fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(
    [
        f"Success ({len(turns_success)} runs)",
        f"Claimed Equivalent ({len(turns_equivalent)} runs)",
        f"Failed ({len(turns_fail)} runs)",
    ]
)
ax.violinplot(
    [
        turns_success,
        turns_equivalent,
        turns_fail,
    ],
    showmeans=True,
)
ax.legend(["Number of Turns"], loc="upper left")
plt.show()


# %% Number of turns

fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
ax.set_xticks(x)
ax.hist(
    data["num_turns"],
    bins=np.arange(20) - 0.5,
    rwidth=0.8,
)
ax.legend(["Number of Turns"], loc="upper left")
plt.show()

# %% Number of turns per successful / unsuccessful run

turns_success = data["num_turns"][data["mutant_killed"]]
turns_equivalent = data["num_turns"][data["claimed_equivalent"]]
turns_fail = data["num_turns"][(~data["mutant_killed"]) & (~data["claimed_equivalent"])]

fig, ax = plt.subplots(layout="constrained", figsize=(8, 4))
ax.set_xticks(x)
ax.hist(
    [turns_success, turns_equivalent, turns_fail],
    bins=np.arange(20) - 0.5,
    rwidth=1,
)
ax.legend(
    [
        "Number of Turns (Success)",
        "Number of Turns (Claimed Equivalent)",
        "Number of Turns (Failed)",
    ],
    loc="upper right",
)
plt.show()


# %% Mean number of turns

data["num_turns"].mean()


# %% Number of experiments / tests per task

x = np.arange(len(data))
fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
ax.violinplot([data["num_experiments"], data["num_tests"]], positions=[1, 2], showmeans=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(["# Experiments", "# Tests"])
ax.legend(["Number of Experiments / Tests"], loc="upper left")
plt.show()
plt.show()

# %% Success rate

data["mutant_killed"].sum() / data["mutant_killed"].count()

# %% Unkilled mutants

data[data["mutant_killed"] == 0][["long_id"]]

# %% Mean test LOC

data["test_loc"].mean()

# %% Conversations with unparsable messages

data[data["tag.incomplete_response"] > 0]["long_id"]

# %% Equivalence claims

data[data["claimed_equivalent"]]["long_id"]

# %% Sample of equivalences

data[data["claimed_equivalent"]].sample(n=10, random_state=1)["long_id"]

# %% Non compilable experiments / tests

data[["num_invalid_experiments", "num_invalid_tests"]].sum()

# %% Tasks with import errors

data[(data["num_experiment_import_errors"] + data["num_test_import_errors"]) > 0]["long_id"]

# %% Compute outcome for simplicity


SUCCESS = "success"
EQUIVALENT = "equivalent"
FAIL = "fail"


def get_outcome(row):
    if row["mutant_killed"]:
        return SUCCESS
    elif row["claimed_equivalent"]:
        return EQUIVALENT
    else:
        return FAIL


data["outcome"] = data.apply(get_outcome, axis=1).to_frame(name="outcome")


# %% Number of runs

# import plotly.express as px
#
# data["color"] = data["mutant_killed"].map(lambda b: "green" if b else "red")
# fig = px.parallel_categories(data, ["mutant_covered", "exit_code_diff", "output_diff", "outcome"], color="color")
# fig.write_html("/mnt/temp/coverage.html", auto_open=True)

# %% Check for differences in exit code and output in all experiments/tests


def has_exit_code_difference(row):
    for exp_or_test in row["experiments"] + row["tests"]:
        if (correct_result := get_result(exp_or_test, mutant=False)) and (
            mutant_result := get_result(exp_or_test, mutant=True)
        ):
            if correct_result["exitcode"] != mutant_result["exitcode"]:
                return True
    return False


def has_output_difference(row):
    for exp_or_test in row["experiments"] + row["tests"]:
        if (correct_result := get_result(exp_or_test, mutant=False)) and (
            mutant_result := get_result(exp_or_test, mutant=True)
        ):
            correct_output = correct_result["output"].replace(correct_result["cwd"], "")
            mutant_output = mutant_result["output"].replace(mutant_result["cwd"], "")
            if correct_output != mutant_output:
                return True
    return False


data["exit_code_diff"] = data.apply(lambda row: has_exit_code_difference(row), axis=1)
data["output_diff"] = data.apply(lambda row: has_output_difference(row), axis=1)

# %% Number of runs per outcome


def print_num_runs_per_outcome(runs):
    print(f"  Total: {len(runs)}")
    print(f"  Successful: {len(runs[runs["outcome"] == "success"])}")
    print(f"  Equivalent: {len(runs[runs["outcome"] == "equivalent"])}")
    print(f"  Failed: {len(runs[runs["outcome"] == "fail"])}")


print("All runs:")
print_num_runs_per_outcome(data)

print("\nRuns that cover the mutant:")
print_num_runs_per_outcome(data[data["mutant_covered"]])

print("\nRuns with a exitcode difference in any experiment or test:")
print_num_runs_per_outcome(data[data["exit_code_diff"]])

print("\nRuns with a output difference in any experiment or test:")
print_num_runs_per_outcome(data[data["output_diff"]])

# %% Runs that had a difference in output but didn't result in a killing test

data[data["output_diff"] & ~data["mutant_killed"]]["id"]

# %% Mutants that were claimed equivalent but were still killed

len(data[data["mutant_killed"] & data["claimed_equivalent"]])

# %% Number of messages

data[data["mutant_killed"] & data["claimed_equivalent"]][["num_messages"]]

# %% Count messages after equivalence
# def msgs_after_equiv(msgs):
#     return list(dropwhile(lambda msg: msg["tag"] != "claimed_equivalent", msgs))
# data["msgs_after_equiv"] = data["conversation"].map(msgs_after_equiv)


# %% Average number of messages
# data["conversation"].map(lambda msgs: [msg for msg in msgs if msg["role"] == "assistant"]).map(len).mean()

# %% Average number of messages after equivalence claim (including runs without equivalence claim)
# data["msgs_after_equiv"].map(lambda msgs: [msg for msg in msgs if msg["role"] == "assistant"]).map(len).mean()

# %% Mutants that were claimed equivalent but were still killed
# data[data["mutant_killed"] & data["claimed_equivalent"]]["conversation"].map(
#     lambda l: len([msg for msg in l if msg["tag"] == "claimed_equivalent"])
# )

# %% Number of runs with too many messages

len(data[data["num_messages"] > 10])

# %% Number of runs with too many messages that killed the mutant

len(data[(data["num_messages"] > 10) & data["mutant_killed"]])

# %% Max number of messages

data["num_messages"].max()
# %%

data[data["mutant_killed"]]["usage.cost"].mean()
# data.groupby("num_messages")[["usage.cost"]].mean()
# %%

(data["num_messages"] - data["num_turns"]).mean()


# %% Compute Token Usage of first 10 messages

data["usage.prompt_tokens.f10"] = data["conversation"].map(
    lambda c: sum([msg["usage"]["prompt_tokens"] for msg in c if msg["role"] == "assistant"][10:])
)
data["usage.completion_tokens.f10"] = data["conversation"].map(
    lambda c: sum([msg["usage"]["completion_tokens"] for msg in c if msg["role"] == "assistant"][10:])
)
data["usage.cached_tokens.f10"] = data["conversation"].map(
    lambda c: sum([msg["usage"].get("cached_tokens", 0) for msg in c if msg["role"] == "assistant"][10:])
)
data["usage.total_tokens.f10"] = data["usage.prompt_tokens.f10"] + data["usage.completion_tokens.f10"]

# Add cost in $ for gpt-4o-mini
# prompt: $0.150 / 1M input tokens
# cached prompt: $0.075 / 1M input tokens
# completion: $0.600 / 1M input tokens
data["usage.cost.f10"] = (
    ((data["usage.prompt_tokens.f10"] - data["usage.cached_tokens.f10"]) * 0.150)
    + (data["usage.cached_tokens.f10"] * 0.075)
    + (data["usage.completion_tokens.f10"] * 0.600)
) / 1_000_000

# %% Compute Token Usage of last 10 messages

data["usage.prompt_tokens.l10"] = data["conversation"].map(
    lambda c: sum([msg["usage"]["prompt_tokens"] for msg in c if msg["role"] == "assistant"][:10])
)
data["usage.completion_tokens.l10"] = data["conversation"].map(
    lambda c: sum([msg["usage"]["completion_tokens"] for msg in c if msg["role"] == "assistant"][:10])
)
data["usage.cached_tokens.l10"] = data["conversation"].map(
    lambda c: sum([msg["usage"].get("cached_tokens", 0) for msg in c if msg["role"] == "assistant"][:10])
)
data["usage.total_tokens.l10"] = data["usage.prompt_tokens.l10"] + data["usage.completion_tokens.l10"]

# Add cost in $ for gpt-4o-mini
# prompt: $0.150 / 1M input tokens
# cached prompt: $0.075 / 1M input tokens
# completion: $0.600 / 1M input tokens
data["usage.cost.l10"] = (
    ((data["usage.prompt_tokens.l10"] - data["usage.cached_tokens.l10"]) * 0.150)
    + (data["usage.cached_tokens.l10"] * 0.075)
    + (data["usage.completion_tokens.l10"] * 0.600)
) / 1_000_000

# %%

data["usage.cost.l10"].sum()

# %%

data["usage.cost.f10"].sum()
