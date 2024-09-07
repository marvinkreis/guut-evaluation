# %% Imports, constants and helpers

import json
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("./quixbugs_benchmark_v0.1")

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

data = pd.json_normalize(results_json)

# %% Compute token usage

prompt_tokens = [
    sum(msg["usage"]["prompt_tokens"] for msg in result["conversation"] if msg["role"] == "assistant")
    for result in results_json
]
completion_tokens = [
    sum(msg["usage"]["completion_tokens"] for msg in result["conversation"] if msg["role"] == "assistant")
    for result in results_json
]
problem_names = [result["problem"]["name"] for result in results_json]
implementations = [result["implementation"] for result in results_json]

data["usage.prompt_tokens"] = prompt_tokens
data["usage.completion_tokens"] = completion_tokens
data["usage.total_tokens"] = data["usage.prompt_tokens"] + data["usage.completion_tokens"]

# Add cost in $ for gpt4o-mini
# prompt: $0.150 / 1M input tokens
# completion: $0.600 / 1M input tokens
data["usage.cost"] = (data["usage.prompt_tokens"] * 0.150 / 1_000_000) + (
    data["usage.completion_tokens"] * 0.600 / 1_000_000
)

# %% Count messages, observations, experiments, tests

num_turns = [len([msg for msg in result["conversation"] if msg["role"] == "assistant"]) for result in results_json]
num_observations = [
    len([exp for exp in result["experiments"] if exp["kind"] == "observation"]) for result in results_json
]
num_experiments = [
    len([exp for exp in result["experiments"] if exp["kind"] == "experiment"]) for result in results_json
]
num_tests = [len(result["tests"]) for result in results_json]

data["num_turns"] = num_turns
data["num_observations"] = num_observations
data["num_experiments"] = num_experiments
data["num_tests"] = num_tests

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

tag_counts = {
    tag: [sum([msg["tag"] == tag for msg in result["conversation"]]) for result in results_json]
    for tag in relevant_msg_tags
}

for tag_name in relevant_msg_tags:
    data[f"tag.{tag_name}"] = tag_counts[tag_name]

# %% Compute test LOC


def estimate_loc(test):
    if test is None:
        return 0
    return len([line for line in test["code"].splitlines() if line.strip() and not line.strip().startswith("#")])


def find_killing_test(result):
    killing_tests = [test for test in result["tests"] if test["kills_mutant"]]
    return killing_tests[0] if killing_tests else None


test_loc = [estimate_loc(find_killing_test(result)) for result in results_json]

data["test_loc"] = test_loc

# %% Sort data!

data = data.sort_values(["implementation", "problem.name", "timestamp"])

# %% Token usage mean

data.groupby("implementation")[["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]].mean()

# %% Token usage sum

data.groupby("implementation")[["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]].sum()

# %% Computation cost per task

problems = data["problem.name"][data["implementation"] == "loop"]
x = np.arange(len(problems))
fig, ax = plt.subplots(layout="constrained", figsize=(15, 8))
ax.bar(x - 0.2, data["usage.cost"][data["implementation"] == "loop"], 0.4, label="loop")
ax.bar(x + 0.2, data["usage.cost"][data["implementation"] == "baseline"], 0.4, label="baseline")
ax.set_xticks(x, problems)
ax.tick_params(axis="x", rotation=90)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f$"))
ax.legend(loc="upper left")
plt.show()

# %% Number of turns per task

problems = data["problem.name"][data["implementation"] == "loop"]
x = np.arange(len(problems))
fig, ax = plt.subplots(layout="constrained", figsize=(15, 8))
ax.bar(x - 0.2, data["num_turns"][data["implementation"] == "loop"], 0.4, label="loop")
ax.bar(x + 0.2, data["num_turns"][data["implementation"] == "baseline"], 0.4, label="baseline")
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

# %% Success rate

data.groupby("implementation")[["mutant_killed"]].sum() / data.groupby("implementation")[["mutant_killed"]].count()

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

# %% Coverage
