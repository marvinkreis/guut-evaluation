# %% Imports, constants and helpers

import json
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(
    "/home/marvin/workspace/guut-evaluation/quixbugs_benchmark_v0.2/04_alt_exp_example_separate"
)

pd.set_option("display.width", 120)

# %% Load json files

results_json = []
for root, dirs, files in RESULTS_DIR.walk():
    result_paths = (root / name for name in files if name == "result.json")
    for path in result_paths:
        # exclude sieve, since it's used as an example in the prompt
        if "sieve" in str(path):
            continue
        result = json.loads(path.read_text())
        if "baseline" in str(root):
            result["implementation"] = "baseline"
        else:
            result["implementation"] = "loop"
        results_json.append(result)

# Sort!
results_json.sort(
    key=lambda result: (
        result["implementation"],
        result["problem"]["name"],
        result["id"],
    )
)

data = pd.json_normalize(results_json)

# %% Compute token usage

data["usage.prompt_tokens"] = data["conversation"].map(
    lambda c: sum(
        msg["usage"]["prompt_tokens"] for msg in c if msg["role"] == "assistant"
    )
)
data["usage.completion_tokens"] = data["conversation"].map(
    lambda c: sum(
        msg["usage"]["completion_tokens"] for msg in c if msg["role"] == "assistant"
    )
)
data["usage.total_tokens"] = (
    data["usage.prompt_tokens"] + data["usage.completion_tokens"]
)

# Add cost in $ for gpt4o-mini
# prompt: $0.150 / 1M input tokens
# completion: $0.600 / 1M input tokens
data["usage.cost"] = (data["usage.prompt_tokens"] * 0.150 / 1_000_000) + (
    data["usage.completion_tokens"] * 0.600 / 1_000_000
)

# %% Count messages, observations, experiments, tests

data["num_turns"] = data["conversation"].map(
    lambda conv: len([msg for msg in conv if msg["role"] == "assistant"])
)
data["num_observations"] = data["experiments"].map(
    lambda exps: len([exp for exp in exps if exp["kind"] == "observation"])
)
data["num_experiments"] = data["experiments"].map(
    lambda exps: len([exp for exp in exps if exp["kind"] == "experiment"])
)
data["num_tests"] = data["tests"].map(len)
data["num_invalid_observations"] = data["experiments"].map(
    lambda exps: len(
        [
            exp
            for exp in exps
            if exp["kind"] == "observation" and not exp["validation_result"]["valid"]
        ]
    )
)
data["num_invalid_experiments"] = data["experiments"].map(
    lambda exps: len(
        [
            exp
            for exp in exps
            if exp["kind"] == "experiment" and not exp["validation_result"]["valid"]
        ]
    )
)
data["num_invalid_tests"] = data["tests"].map(
    lambda tests: len(
        [test for test in tests if not test["validation_result"]["valid"]]
    )
)

# %% Compute counts of message types

relevant_msg_tags = [
    "experiment_stated",
    "experiment_doesnt_compile",
    "experiment_results_given",
    "test_instructions_given",
    "test_stated",
    "test_invalid",
    "test_doesnt_detect_mutant",
    "claimed_equivalent",
    "done",
    "incomplete_response",
    "aborted",
]

for tag in relevant_msg_tags:
    data[f"tag.{tag}"] = data["conversation"].map(
        lambda conv: len([msg for msg in conv if msg["tag"] == tag])
    )

# %% Compute test LOC


def estimate_loc(test):
    if test is None:
        return None
    return len(
        [
            line
            for line in test["code"].splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    )


def find_killing_test(tests):
    killing_tests = [test for test in tests if test["kills_mutant"]]
    return killing_tests[0] if killing_tests else None


data["test_loc"] = data["tests"].map(
    lambda tests: estimate_loc(find_killing_test(tests))
)

# %% Compute number of import errors


def count_import_errors(exps_or_tests):
    return len(
        [
            exp
            for exp in exps_or_tests
            if exp["result"]
            and "ModuleNotFoundError"
            in (exp["result"].get("test") or exp["result"].get("correct"))["output"]
        ]
    )


data["num_observation_import_errors"] = data["experiments"].map(
    lambda exps: count_import_errors(
        [exp for exp in exps if exp["kind"] == "observation"]
    )
)
data["num_experiment_import_errors"] = data["experiments"].map(
    lambda exps: count_import_errors(
        [exp for exp in exps if exp["kind"] == "experiment"]
    )
)
data["num_test_import_errors"] = data["tests"].map(
    lambda tests: count_import_errors(tests)
)

# %% Count number of debugger scripts

data["num_debugger_scripts"] = data["experiments"].map(
    lambda exps: len([exp for exp in exps if exp["debugger_script"]])
)

# %% Percentage of experiments with debugger scripts

loop_data = data[data["implementation"] == "loop"]
loop_data["num_debugger_scripts"].sum() / (
    loop_data["num_experiments"].sum() + loop_data["num_observations"].sum()
)

# %% Token usage mean

data.groupby("implementation")[
    ["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]
].mean()

# %% Token usage sum

data.groupby("implementation")[
    ["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]
].sum()

# %% Computation cost per task

problems = data["problem.name"][data["implementation"] == "loop"]
x = np.arange(len(problems))
fig, ax = plt.subplots(layout="constrained", figsize=(15, 8))
ax.bar(x - 0.2, data["usage.cost"][data["implementation"] == "loop"], 0.4, label="loop")
ax.set_xticks(x, problems)
ax.tick_params(axis="x", rotation=90)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f$"))
ax.legend(loc="upper left")
plt.show()

# %% Mean number of turns per task

data.groupby("implementation")["num_turns"].mean()

# %% Number of turns per task

problems = data["problem.name"][data["implementation"] == "loop"]
x = np.arange(len(problems))
fig, ax = plt.subplots(layout="constrained", figsize=(15, 8))
ax.bar(x - 0.2, data["num_turns"][data["implementation"] == "loop"], 0.4, label="loop")
# ax.bar(
#     x + 0.2,
#     data["num_turns"][data["implementation"] == "baseline"],
#     0.4,
#     label="baseline",
# )
ax.set_xticks(x, problems)
ax.tick_params(axis="x", rotation=90)
ax.legend(loc="upper left")
plt.show()

# %% Number of observations / experiments / tests per task (loop only)

data[data["implementation"] == "loop"].plot.bar(
    x="problem.name",
    y=["num_observations", "num_experiments", "num_tests"],
    stacked=True,
    layout="constrained",
    figsize=(15, 8),
)
plt.show()

# %% Number of observations / experiments / tests in total

data.groupby("implementation")[
    ["num_observations", "num_experiments", "num_tests"]
].sum()

# %% Success rate

data.groupby("implementation")[["mutant_killed"]].sum() / data.groupby(
    "implementation"
)[["mutant_killed"]].count()

# %% Unkilled mutants

data[data["mutant_killed"] == 0][["implementation", "problem.name", "id"]]

# %% Mean test LOC

data.groupby("implementation")[["test_loc"]].mean()

# %% Conversations with unparsable messages

data[data["tag.incomplete_response"] > 0][["implementation", "problem.name", "id"]]

# %% Aborted conversations

data[data["tag.aborted"] > 0][["implementation", "problem.name", "id"]]

# %% Equivalence claims

data[data["tag.claimed_equivalent"] > 0][["implementation", "problem.name", "id"]]

# %% Non compilable experiments / tests

data.groupby("implementation")[
    ["num_invalid_observations", "num_invalid_experiments", "num_invalid_tests"]
].sum()

# %% Tasks with import errors

data[
    (
        data["num_observation_import_errors"]
        + data["num_experiment_import_errors"]
        + data["num_test_import_errors"]
    )
    > 0
][["implementation", "problem.name", "id", "num_test_import_errors"]]

# %% Coverage


def find_killing_or_last_test(tests):
    killing_tests = [test for test in tests if test["kills_mutant"]]
    if killing_tests:
        return killing_tests[-1]
    if tests:
        return tests[-1]
    return None


def get_coverage_from_test(test):
    if test is None:
        return None, None
    test_result = test.get("result")
    if test_result is None:
        return None, None
    coverage = test_result["correct"].get("coverage")
    if coverage is None:
        return None, None
    return len(coverage["covered_lines"]), len(coverage["missing_lines"])


data["coverage.covered"] = data["tests"].map(
    lambda tests: get_coverage_from_test(find_killing_test(tests))[0]
)
data["coverage.missing"] = data["tests"].map(
    lambda tests: get_coverage_from_test(find_killing_test(tests))[1]
)
data["coverage"] = data["coverage.covered"] / (
    data["coverage.missing"] + data["coverage.covered"]
)

# %% Mean coverage (loop)

data["coverage"][data["coverage"] != np.nan][data["implementation"] == "loop"].mean()

# %% Tasks with less than 100% coverage

data[data["coverage"] < 1.0][
    ["id", "coverage.missing", "coverage.covered", "coverage", "mutant_killed"]
]
