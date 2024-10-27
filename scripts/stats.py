# %% Imports, constants and helpers

import json
import os
import sqlite3
from functools import partial, reduce
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple
from matplotlib import colormaps as cm

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

JsonObj = Dict[str, Any]

# %% Find result directory and mutants

if "get_ipython" in locals():
    print("Running as ipython kernel")
    RESULTS_DIR = Path(os.getcwd()).parent / "guut_emse_results"
    MUTANTS_DIR = Path(os.getcwd()).parent / "emse_projects_data" / "mutants_sampled"
else:
    print("Running as script")
    RESULTS_DIR = Path(__file__).parent / "guut_emse_results"
    MUTANTS_DIR = Path(__file__).parent / "emse_projects_data" / "mutants_sampled"


# %% Load json result files

if "data" in locals():
    raise Exception("data is already in memory. Refusing to read the files again.")


def prepare_loop_result(loop_result: JsonObj):
    """Delete some unnecessary data to save memory."""

    # Prepare tests
    for test in loop_result["tests"]:
        # Clean execution results
        if (result := test["result"]) is not None:
            for exec_result in [result["correct"], result["mutant"]]:
                if coverage := exec_result["coverage"]:
                    del coverage["raw"]

    # Prepare experiments
    for experiment in loop_result["experiments"]:
        # Clean execution results
        if (result := experiment["result"]) is not None:
            for exec_result in [
                result["test_correct"],
                result["test_mutant"],
                result["debug_correct"],
                result["debug_mutant"],
            ]:
                if exec_result is None:
                    continue
                if coverage := exec_result["coverage"]:
                    del coverage["raw"]

    # Prepare messages
    for msg in loop_result["conversation"]:
        del msg["content"]


def read_result_full(path: Path) -> JsonObj:
    with path.open("r") as f:
        result = json.load(f)
    prepare_loop_result(result)
    return result


print("Reading tons of json files. This may take a while...")
with Pool(8) as pool:
    results_json = pool.imap_unordered(read_result_full, RESULTS_DIR.glob("*/loops/*/result.json"), chunksize=100)
    data = pd.json_normalize(results_json)
    data = data.sort_values(by=["long_id", "problem.target_path", "problem.mutant_op", "problem.occurrence"])
    del results_json
print("done")

# %% Print memory usage

with open("/proc/self/status") as f:
    memusage = f.read().split("VmRSS:")[1].split("\n")[0][:-3]
    print(f"Memory used: {int(memusage.strip()) / (1024**2):.3f} GB")

# %% Add project name to data

package_to_project = {
    "apimd": "apimd",
    "black": "black",
    "blackd": "black",
    "black_primer": "black",
    "blib2to3": "black",
    "_black_version": "black",
    "codetiming": "codetiming",
    "dataclasses_json": "dataclasses-json",
    "docstring_parser": "docstring_parser",
    "flake8": "flake8",
    "flutes": "flutes",
    "flutils": "flutils",
    "httpie": "httpie",
    "isort": "isort",
    "mimesis": "mimesis",
    "pdir": "pdir2",
    "py_backwards": "py-backwards",
    "pymonet": "pyMonet",
    "pypara": "pypara",
    "semantic_release": "python-semantic-release",
    "string_utils": "python-string-utils",
    "pytutils": "pytutils",
    "sanic": "sanic",
    "sty": "sty",
    "thonny": "thonny",
    "typesystem": "typesystem",
}
data["project"] = data["problem.module_path"].map(lambda path: package_to_project[path.rsplit("/")[-1]])

# %% Add lines of mutants


def read_mutant_lines(mutants_file: Path) -> Dict[Tuple[str, str, str], Tuple[int, int]]:
    con = sqlite3.connect(mutants_file)
    cur = con.execute("select module_path, operator_name, occurrence, start_pos_row, end_pos_row from mutation_specs;")
    mutants = cur.fetchall()
    con.close()
    return {(m[0], m[1], m[2]): (m[3], m[4]) for m in mutants}


mutant_lines = {project: read_mutant_lines(MUTANTS_DIR / f"{project}.sqlite") for project in data["project"].unique()}
data["mutant_lines"] = data.apply(
    lambda row: mutant_lines[row["project"]][
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
del mutant_lines


# %% Helper functions


def get_result(exp_or_test: JsonObj, mutant: bool = False) -> JsonObj | None:
    if exp_or_test is None:
        return None

    result = exp_or_test.get("result")
    if result is None:
        return None

    if not mutant:
        return result.get("test_correct") or result.get("correct")
    else:
        return result.get("test_mutant") or result.get("mutant")


def find_killing_test(tests: List[JsonObj]) -> JsonObj | None:
    killing_tests = [test for test in tests if test["kills_mutant"]]
    return killing_tests[0] if killing_tests else None


# %% Strings to colors


cmap = cm.get_cmap("tab10")


def string_colors(strings: Iterable[str], num_colors=10):
    known_strings = {}
    current_color = 0
    colors = []
    for string in strings:
        if color := known_strings.get(string):
            colors.append(color)
        else:
            known_strings[string] = current_color
            colors.append(current_color)
            current_color = (current_color + 1) % num_colors
    return colors


# %% Check whether each run covers the mutant. This is not very accurate.


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
data["coverage.covered_line_ids"] = data.apply(
    lambda row: [f'{row["problem.target_path"]}::{line}' for line in row["coverage.covered_lines"]], axis=1
)
data["coverage.missing_line_ids"] = data.apply(
    lambda row: [f'{row["problem.target_path"]}::{line}' for line in row["coverage.missing_lines"]], axis=1
)


# %% Count number of coverable lines and number of mutants per module

all_lines = {}
all_line_ids = {}
num_mutants = {}

for project in data["project"].unique():
    for target in data[data["project"] == project]["problem.target_path"].unique():
        covered = data[
            (data["project"] == project) & (data["problem.target_path"] == target) & (data["coverage.covered_lines"])
        ]
        if len(covered):
            all_lines[(project, target)] = (
                covered.iloc[0]["coverage.covered_lines"] + covered.iloc[0]["coverage.missing_lines"]
            )
            all_line_ids[(project, target)] = (
                covered.iloc[0]["coverage.covered_line_ids"] + covered.iloc[0]["coverage.missing_line_ids"]
            )

        num_mutants[(project, target)] = len(
            data[
                (data["project"] == project)
                & (data["problem.target_path"] == target)
                & (data["settings.preset_name"] == "debugging_one_shot")
            ]
        )

data["num_mutants_in_module"] = data.apply(
    lambda row: num_mutants[(row["project"], row["problem.target_path"])], axis=1
)
data["coverage.all_lines"] = data.apply(
    lambda row: all_lines.get((row["project"], row["problem.target_path"]), []), axis=1
)
data["coverage.num_lines"] = data["coverage.all_lines"].map(len)
data["coverage.all_line_ids"] = data.apply(
    lambda row: all_line_ids.get((row["project"], row["problem.target_path"]), []), axis=1
)

del num_mutants
del all_lines
del all_line_ids


# %% Plot Coverage


def add_coverage(x: List[str], y: List[str]) -> List[str]:
    acc = set(x)
    acc.update(y)
    return list(acc)


add_coverage_np = np.frompyfunc(add_coverage, 2, 1)


for project in np.unique(data[["project"]]):
    fig, ax = plt.subplots(layout="constrained", figsize=(8, 4))
    for preset in [
        "baseline_without_iterations",
        "baseline_with_iterations",
        "debugging_zero_shot",
        "debugging_one_shot",
    ]:
        target_data = data[(data["project"] == project) & (data["settings.preset_name"] == preset)]
        target_data = target_data.sort_values(by=["num_mutants_in_module", "problem.module_path"], ascending=False)

        acc_coverage = add_coverage_np.accumulate(target_data["coverage.covered_line_ids"])
        all_lines = reduce(add_coverage, target_data["coverage.all_line_ids"], [])
        acc_coverage_percent = acc_coverage.map(lambda c: len(c) / max(len(all_lines), 1))

        x = np.arange(len(acc_coverage_percent))
        y = acc_coverage_percent

        ax.plot(x, acc_coverage_percent, 0.6, label=f"{preset}")

    current_target = ""
    lines = []
    for index, (_, row) in enumerate(target_data.iterrows()):
        new_target = row["problem.target_path"]
        if new_target != current_target:
            current_target = new_target
            lines.append(index)
    ax.vlines(x=lines, ymin=0.0, ymax=1.0, colors="lightgrey", ls="--", lw=1, label="modules")

    fig.text(0.00, 1.02, project)
    ax.legend()
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

data["num_messages.l10"] = data["conversation"].map(
    lambda c: len([msg for msg in c if msg["role"] == "assistant"][10:])
)
data["usage.prompt_tokens.l10"] = data["conversation"].map(
    lambda c: sum([msg["usage"]["prompt_tokens"] for msg in c if msg["role"] == "assistant"][10:])
)
data["usage.completion_tokens.l10"] = data["conversation"].map(
    lambda c: sum([msg["usage"]["completion_tokens"] for msg in c if msg["role"] == "assistant"][10:])
)
data["usage.cached_tokens.l10"] = data["conversation"].map(
    lambda c: sum([msg["usage"].get("cached_tokens", 0) for msg in c if msg["role"] == "assistant"][10:])
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
# %% Compute Token Usage of last 10 messages

data["num_messages.f10"] = data["conversation"].map(
    lambda c: len([msg for msg in c if msg["role"] == "assistant"][:10])
)
data["usage.prompt_tokens.f10"] = data["conversation"].map(
    lambda c: sum([msg["usage"]["prompt_tokens"] for msg in c if msg["role"] == "assistant"][:10])
)
data["usage.completion_tokens.f10"] = data["conversation"].map(
    lambda c: sum([msg["usage"]["completion_tokens"] for msg in c if msg["role"] == "assistant"][:10])
)
data["usage.cached_tokens.f10"] = data["conversation"].map(
    lambda c: sum([msg["usage"].get("cached_tokens", 0) for msg in c if msg["role"] == "assistant"][:10])
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

# %%

data["num_messages.f10"].sum()

# %%

data["num_messages.l10"].sum()

# %%

data["usage.cost.l10"].sum()

# %%

data["usage.cost.f10"].sum()
