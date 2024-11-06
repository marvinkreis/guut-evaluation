# %% Imports and constants

import json
import os
import sqlite3
from functools import partial, reduce
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, NamedTuple, Tuple, cast
from matplotlib import colormaps as cm
from tqdm import tqdm
from math import floor

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: f"{int(x):d}" if floor(x) == x else f"{x:.3f}")

JsonObj = Dict[str, Any]
OUTPUT_PATH = Path("/tmp/out")
OUTPUT_PATH.mkdir(exist_ok=True)


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

    module_name = Path(loop_result["problem"]["module_path"]).name
    target_path = loop_result["problem"]["target_path"]

    # Prepare tests
    for test in loop_result["tests"]:
        # Clean execution results
        if (result := test["result"]) is not None:
            for exec_result in [result["correct"], result["mutant"]]:
                # If coverage is present, parse branch coverage and delete the raw coverage to save memory
                if coverage := exec_result["coverage"]:
                    if raw_file_coverage := coverage["raw"]["files"][f"{module_name}/{target_path}"]:
                        coverage["covered_branches"] = raw_file_coverage["executed_branches"]
                        coverage["missing_branches"] = raw_file_coverage["missing_branches"]
                    else:
                        coverage["covered_branches"] = []
                        coverage["missing_branches"] = []
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
                # If coverage is present, parse branch coverage and delete the raw coverage to save memory
                if coverage := exec_result["coverage"]:
                    if raw_file_coverage := coverage["raw"]["files"][f"{module_name}/{target_path}"]:
                        coverage["covered_branches"] = raw_file_coverage["executed_branches"]
                        coverage["missed_branches"] = raw_file_coverage["missing_branches"]
                    else:
                        coverage["covered_branches"] = []
                        coverage["missed_branches"] = []
                    del coverage["raw"]

    # Prepare messages
    for msg in loop_result["conversation"]:
        del msg["content"]


def read_result_full(path: Path) -> JsonObj:
    with path.open("r") as f:
        result = json.load(f)
    prepare_loop_result(result)
    result["path"] = str(path)
    return result


results_paths = list(RESULTS_DIR.glob("*/loops/*/result.json"))
with Pool(8) as pool:
    results_json = list(
        tqdm(
            pool.imap_unordered(read_result_full, results_paths, chunksize=100),
            total=len(results_paths),
            desc="Loading tons of json files",
        )
    )

data = pd.json_normalize(results_json)
data = data.sort_values(by=["long_id", "problem.target_path", "problem.mutant_op", "problem.occurrence"])
del results_json


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


# %% Add mutant identifier to data


class MutantId(NamedTuple):
    project: str
    target_path: str
    mutant_op: str
    occurrence: int


def mutant_id_from_row(row):
    return MutantId(
        project=row["project"],
        target_path=row["problem.target_path"],
        mutant_op=row["problem.mutant_op"],
        occurrence=row["problem.occurrence"],
    )


data["mutant_id"] = data.apply(mutant_id_from_row, axis=1)

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


# %% Make preset easier to type

data["preset"] = data["settings.preset_name"]


# %% Add lines of mutants


def read_mutant_lines(mutants_file: Path, project: str) -> Dict[MutantId, Tuple[int, int]]:
    con = sqlite3.connect(mutants_file)
    cur = con.execute("""
        select
            module_path,
            operator_name,
            occurrence,
            start_pos_row,
            end_pos_row
        from mutation_specs;
    """)
    mutants = cur.fetchall()
    con.close()
    return {MutantId(project, m[0], m[1], m[2]): (m[3], m[4]) for m in mutants}


mutant_lines = {}
for project in data["project"].unique():
    mutant_lines.update(read_mutant_lines(MUTANTS_DIR / f"{project}.sqlite", project))

data["mutant_lines"] = data["mutant_id"].map(mutant_lines.get)
data["mutant_lines.start"] = data["mutant_lines"].map(lambda t: t[0])
data["mutant_lines.end"] = data["mutant_lines"].map(lambda t: t[1])
del mutant_lines


# %% Add cosmic-ray results


def test_outcome_kills(cosmic_ray_outcome):
    if cosmic_ray_outcome not in ["SURVIVED", "KILLED"]:
        raise Exception(f"Unexpected test outcome: {cosmic_ray_outcome}")
    return cosmic_ray_outcome == "KILLED"


def read_mutant_results(mutants_file: Path, project: str) -> Dict[MutantId, str]:
    con = sqlite3.connect(mutants_file)
    cur = con.execute("""
        select
            module_path,
            operator_name,
            occurrence,
            test_outcome
        from mutation_specs
        left join work_results
            on mutation_specs.job_id = work_results.job_id;
    """)
    mutants = cur.fetchall()
    con.close()
    return {MutantId(project, m[0], m[1], m[2]): test_outcome_kills(m[3]) for m in mutants}


cosmic_ray_results = {}
# Maps (project, preset) to a map that maps (target_path, mutant_op, occurrence) to mutant test result
for mutants_path in RESULTS_DIR.glob("*/cosmic-ray/mutants.sqlite"):
    results_dir = (mutants_path / ".." / "..").resolve()
    parts = results_dir.name.split("_")
    preset = "_".join(parts[:3])
    package = "_".join(parts[3:-1])
    project = package_to_project[package]
    cosmic_ray_results[(project, preset)] = read_mutant_results(mutants_path, project)


data["cosmic_ray.killed_by_own"] = data.apply(
    lambda row: cosmic_ray_results[(row["project"], row["preset"])][mutant_id_from_row(row)],
    axis=1,
)
presets = list(data["preset"].unique())
data["cosmic_ray.killed_by_any"] = data.apply(
    lambda row: any(cosmic_ray_results[(row["project"], preset)][mutant_id_from_row(row)] for preset in presets),
    axis=1,
)

del cosmic_ray_results


# %% Check which mutants are claimed equivalent by multiple runs


equivalence_claims = {}
for index, row in data.iterrows():
    mutant_id = mutant_id_from_row(row)
    equivalence_claims[mutant_id] = equivalence_claims.get(mutant_id, 0) + int(row["claimed_equivalent"])

presets = list(data["preset"].unique())
data["mutant.num_equivalence_claims"] = data.apply(
    lambda row: equivalence_claims[mutant_id_from_row(row)],
    axis=1,
)
del equivalence_claims


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


# %% Check whether each run covers the mutant (via line coverage). This is not very accurate.


def covers_mutant(row, experiments_or_tests: Literal["experiments", "tests"]):
    start, end = row["mutant_lines"]
    mutant_lines = set(range(start, end + 1))
    return any(
        set(coverage["covered_lines"]).intersection(mutant_lines)
        for exp in row[experiments_or_tests]
        if (exec_result := get_result(exp)) and (coverage := exec_result["coverage"])
    )


data["mutant_covered.by_experiment"] = data.apply(partial(covers_mutant, experiments_or_tests="experiments"), axis=1)
data["mutant_covered.by_test"] = data.apply(partial(covers_mutant, experiments_or_tests="tests"), axis=1)
data["mutant_covered"] = data["mutant_covered.by_experiment"] | data["mutant_covered.by_test"]


# %% Get line coverage from killing tests


def get_line_coverage_from_test(test):
    exec_result = get_result(test)
    if (exec_result is None) or (exec_result["coverage"] is None):
        return [], []
    coverage = exec_result["coverage"]
    return coverage["covered_lines"], coverage["missing_lines"]


data["coverage.covered_lines"] = data["tests"].map(
    lambda tests: get_line_coverage_from_test(find_killing_test(tests))[0]
)
data["coverage.missing_lines"] = data["tests"].map(
    lambda tests: get_line_coverage_from_test(find_killing_test(tests))[1]
)
data["coverage.covered_line_ids"] = data.apply(
    lambda row: [f'{row["problem.target_path"]}::{line}' for line in row["coverage.covered_lines"]], axis=1
)
data["coverage.missing_line_ids"] = data.apply(
    lambda row: [f'{row["problem.target_path"]}::{line}' for line in row["coverage.missing_lines"]], axis=1
)


# %% Get branch coverage from killing tests


def get_branch_coverage_from_test(test):
    exec_result = get_result(test)
    if (exec_result is None) or (exec_result["coverage"] is None):
        return [], []
    coverage = exec_result["coverage"]
    return coverage["covered_branches"], coverage["missing_branches"]


data["coverage.covered_branches"] = data["tests"].map(
    lambda tests: get_branch_coverage_from_test(find_killing_test(tests))[0]
)
data["coverage.missing_branches"] = data["tests"].map(
    lambda tests: get_branch_coverage_from_test(find_killing_test(tests))[1]
)
data["coverage.covered_branch_ids"] = data.apply(
    lambda row: [f'{row["problem.target_path"]}::{branch}' for branch in row["coverage.covered_branches"]], axis=1
)
data["coverage.missing_branch_ids"] = data.apply(
    lambda row: [f'{row["problem.target_path"]}::{branch}' for branch in row["coverage.missing_branches"]], axis=1
)


# %% Count number of mutants per module

num_mutants = {}

for project in data["project"].unique():
    for target_path in set(data[data["project"] == project]["problem.target_path"]):
        num_mutants[(project, target_path)] = len(
            data[
                (data["project"] == project)
                & (data["problem.target_path"] == target_path)
                & (data["preset"] == "debugging_one_shot")
            ]
        )

data["module.num_mutants"] = data.apply(lambda row: num_mutants[(row["project"], row["problem.target_path"])], axis=1)
del num_mutants


# %% Get number of coverable lines and branches per file

all_lines = {}
all_line_ids = {}
all_branches = {}
all_branch_ids = {}

for project in data["project"].unique():
    for target_path in set(data[data["project"] == project]["problem.target_path"]):
        covered = data[
            (data["project"] == project)
            & (data["problem.target_path"] == target_path)
            & (data["coverage.covered_lines"])
        ]
        if len(covered):
            all_lines[(project, target_path)] = (
                covered.iloc[0]["coverage.covered_lines"] + covered.iloc[0]["coverage.missing_lines"]
            )
            all_line_ids[(project, target_path)] = (
                covered.iloc[0]["coverage.covered_line_ids"] + covered.iloc[0]["coverage.missing_line_ids"]
            )
            all_branches[(project, target_path)] = (
                covered.iloc[0]["coverage.covered_branches"] + covered.iloc[0]["coverage.missing_branches"]
            )
            all_branch_ids[(project, target_path)] = (
                covered.iloc[0]["coverage.covered_branch_ids"] + covered.iloc[0]["coverage.missing_branch_ids"]
            )
        else:
            print(f"No coverage data available for {project} {target_path}")

data["coverage.all_lines"] = data.apply(
    lambda row: all_lines.get((row["project"], row["problem.target_path"]), []), axis=1
)
data["coverage.all_line_ids"] = data.apply(
    lambda row: all_line_ids.get((row["project"], row["problem.target_path"]), []), axis=1
)
data["coverage.num_lines"] = data["coverage.all_lines"].map(len)

data["coverage.all_branches"] = data.apply(
    lambda row: all_branches.get((row["project"], row["problem.target_path"]), []), axis=1
)
data["coverage.all_branch_ids"] = data.apply(
    lambda row: all_branch_ids.get((row["project"], row["problem.target_path"]), []), axis=1
)
data["coverage.num_branches"] = data["coverage.all_branches"].map(len)

del all_lines
del all_line_ids
del all_branches
del all_branch_ids


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


# %% Compute Token Usage of first 10 messages

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


# %% Compute Token Usage of last 10 messages

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


# %% Count messages, experiments, tests

data["num_completions"] = data["conversation"].map(
    lambda conv: len([msg for msg in conv if msg["role"] == "assistant"])
)
data["num_experiments"] = data["experiments"].map(len)
data["num_tests"] = data["tests"].map(len)
data["num_turns"] = data["num_experiments"] + data["num_tests"]

data["num_invalid_experiments"] = data["experiments"].map(
    lambda exps: len([exp for exp in exps if not exp["validation_result"]["valid"]])
)
data["num_invalid_tests"] = data["tests"].map(
    lambda tests: len([test for test in tests if not test["validation_result"]["valid"]])
)


# %% Count invalid messages

data["num_incomplete_responses"] = data["conversation"].map(
    lambda conv: len([msg for msg in conv if msg["tag"] == "incomplete_response"])
)

# %% Estimate test LOC


def estimate_loc(test):
    if test is None:
        return None
    return len([line for line in test["code"].splitlines() if line.strip() and not line.strip().startswith("#")])


data["test_loc"] = data["tests"].map(lambda tests: estimate_loc(find_killing_test(tests)))


# %% Count import errors


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

# %% Print available columns

cols = list(data.columns)
mid = int(((len(cols) + 1) // 2))
for x, y in zip(cols[:mid] + [""], cols[mid:] + [""]):
    print(f"{x:<40} {y}")


# %% Strings to colors


cmap = cm.get_cmap("tab20")
cmap_colors = cmap.colors  # pyright: ignore


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


# %% Plot Coverage


USE_BRANCH_COVERAGE = True


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
        target_data = cast(pd.DataFrame, data[(data["project"] == project) & (data["preset"] == preset)])
        target_data = target_data.sort_values(by=["module.num_mutants", "problem.target_path"], ascending=False)

        if USE_BRANCH_COVERAGE:
            acc_coverage = add_coverage_np.accumulate(target_data["coverage.covered_branch_ids"])
            all_coverable_objects = reduce(add_coverage, target_data["coverage.all_branch_ids"], [])
        else:
            acc_coverage = add_coverage_np.accumulate(target_data["coverage.covered_line_ids"])
            all_coverable_objects = reduce(add_coverage, target_data["coverage.all_line_ids"], [])

        coverage_percent = np.vectorize(lambda c: len(c) / max(len(all_coverable_objects), 1))
        acc_coverage_percent = coverage_percent(acc_coverage)

        x = np.arange(len(acc_coverage_percent))
        y = acc_coverage_percent

        ax.plot(x, acc_coverage_percent, 0.6, label=f"{preset}")

        if preset == "debugging_one_shot":  # arbitrary preset, just run this once
            current_target = ""
            lines = []
            for index, new_target in enumerate(target_data["problem.target_path"]):
                if new_target != current_target:
                    current_target = new_target
                    lines.append(index)
            ax.vlines(x=lines, ymin=0.0, ymax=1.0, colors="lightgrey", ls="--", lw=1, label="modules")

    fig.text(0.00, 1.02, project)
    ax.legend()
    plt.show()


# %% Token usage mean for a single mutant

data.groupby("preset")[["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]].mean()
# %% Token usage mean for a single project (1000 or less mutants)

data.groupby(["preset", "project"])[["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]].sum().groupby(
    "preset"
).mean()
# %% Token usage for each single project (1000 or less mutants)

data.groupby(["preset", "project"])[["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]].sum()

# %% Token usage sum

data[["usage.prompt_tokens", "usage.completion_tokens", "usage.cost"]].sum()

# %% Number of turns

target_data = {preset: data[data["preset"] == preset] for preset in data["preset"].unique()}

fig, ax = plt.subplots(layout="constrained", figsize=(10, 6))
ax.set_xticks(np.arange(len(target_data)) + 1)
ax.set_xticklabels([preset for preset in target_data.keys()])
ax.violinplot([preset_data["num_turns"] for preset_data in target_data.values()], showmeans=True)
ax.legend(["Number of Turns"], loc="upper left")
plt.show()


# %% Number of turns per successful / unsuccessful run

for preset in data["preset"].unique():
    target_data = data[data["preset"] == preset]

    turns_success = target_data["num_turns"][target_data["outcome"] == SUCCESS]
    turns_equivalent = target_data["num_turns"][target_data["outcome"] == EQUIVALENT]
    turns_fail = target_data["num_turns"][target_data["outcome"] == FAIL]

    if len(turns_equivalent) == 0:
        turns_equivalent = [0]
        num_equivalent = 0
    else:
        num_equivalent = len(turns_equivalent)

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 6))
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(
        [
            f"Success ({len(turns_success)} runs)",
            f"Claimed Equivalent ({num_equivalent} runs)",
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
    fig.text(0.00, 1.02, preset)
    plt.show()


# %% Number of turns

for preset in data["preset"].unique():
    target_data = data[data["preset"] == preset]

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 4))
    bins = np.arange(11) + 1

    ax.set_xticks(bins)
    ax.hist(
        target_data["num_turns"],
        bins=[float(x) for x in bins],
        rwidth=0.8,
    )
    ax.legend(["Number of Conversations with N Turns"], loc="upper left")
    fig.text(0.00, 1.02, preset)
    plt.show()


# %% Number of turns per successful / unsuccessful run

for preset in data["preset"].unique():
    target_data = data[data["preset"] == preset]

    turns_success = target_data["num_turns"][target_data["outcome"] == SUCCESS]
    turns_equivalent = target_data["num_turns"][target_data["outcome"] == EQUIVALENT]
    turns_fail = target_data["num_turns"][target_data["outcome"] == FAIL]

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 4))
    bins = np.arange(11) + 1

    ax.set_xticks(bins)
    ax.hist(
        [turns_success, turns_equivalent, turns_fail],
        bins=[float(x) for x in bins],
        rwidth=1,
    )
    ax.legend(
        [
            "Number of Conversations with N Turns (Success)",
            "Number of Conversations with N Turns (Claimed Equivalent)",
            "Number of Conversations with N Turns (Failed)",
        ],
        loc="upper right",
    )
    fig.text(0.00, 1.02, preset)
    plt.show()


# %% Number of experiments / tests per mutant

for preset in data["preset"].unique():
    if not preset.startswith("debugging"):
        continue

    target_data = data[data["preset"] == preset]

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
    ax.violinplot([target_data["num_experiments"], target_data["num_tests"]], positions=[1, 2], showmeans=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["# Experiments", "# Tests"])
    ax.legend(["Number of Experiments / Tests"], loc="upper left")
    fig.text(0.00, 1.02, preset)
    plt.show()

# %% Success rate

print(data["mutant_killed"].sum() / data["mutant_killed"].count())

# %% Mean test LOC

data.groupby("preset")["test_loc"].mean()

# %% Conversations with unparsable messages

data[data["num_incomplete_responses"] > 0]["long_id"]

# %% Non compilable experiments / tests

data[["num_invalid_experiments", "num_invalid_tests"]].sum()

# %% Mutants with import errors

len(data[(data["num_experiment_import_errors"] + data["num_test_import_errors"]) > 0])


# %% Number of runs

# import plotly.express as px
#
# data["color"] = data["mutant_killed"].map(lambda b: "green" if b else "red")
# fig = px.parallel_categories(data, ["mutant_covered", "exit_code_diff", "output_diff", "outcome"], color="color")
# fig.write_html("/mnt/temp/coverage.html", auto_open=True)


# %% Number of runs per outcome


def print_num_runs_per_outcome(runs):
    print(f"  Total: {len(runs)}")
    print(f"  Successful: {len(runs[runs["outcome"] == "success"])}")
    print(f"  Equivalent: {len(runs[runs["outcome"] == "equivalent"])}")
    print(f"  Failed: {len(runs[runs["outcome"] == "fail"])}")


for preset in data["preset"].unique():
    target_data = data[data["preset"] == preset]
    print(f"{preset}:")

    print("All runs:")
    print_num_runs_per_outcome(target_data)

    print("\nRuns that cover the mutant:")
    print_num_runs_per_outcome(target_data[target_data["mutant_covered"]])

    print("\nRuns that detected a output difference in any experiment or test:")
    print_num_runs_per_outcome(target_data[target_data["output_diff"]])

    print("\nRuns that detected an exitcode difference in any experiment or test:")
    print_num_runs_per_outcome(target_data[target_data["exit_code_diff"]])

    print("\n")

# %% Runs that had a difference in output but didn't result in a killing test

len(data[data["output_diff"] & ~data["mutant_killed"]]["id"])


# %% Mutants that were claimed equivalent but were still killed

len(data[data["mutant_killed"] & data["claimed_equivalent"]])


# %% Number of runs with more completions than allowed turns

len(data[data["num_completions"] > 10])

# %% Number of runs with more completions than allowed turns, that still killed the mutant

len(data[(data["num_completions"] > 10) & data["mutant_killed"]])

# %% Equivalences per project

equivs = data[data["claimed_equivalent"]]
equivs.groupby(["project", "preset"]).size()


# %% Failed runs with less than 10 messages

data[(data["outcome"] == FAIL) & (data["num_turns"] != 10) & (data["preset"] != "baseline_without_iterations")][
    ["long_id", "num_completions"]
]


# %% Sample of equivalences

data[data["claimed_equivalent"]].sample(n=10, random_state=1)["long_id"]


# %% Killed mutants

print(f"Directly killed mutants: {len(data[data["mutant_killed"]])}")
print(f"Mutants killed by tests from the same run: {len(data[data["cosmic_ray.killed_by_own"]])}")
print(f"Mutants killed by tests from the any run: {len(data[data["cosmic_ray.killed_by_any"]])}")


# %% Equivalent mutants

print(f"Equivalence-claimed mutants: {len(data[data["claimed_equivalent"]])}")
print(
    f"Equivalence-claimed mutants not killed by test from same run: {len(data[data["claimed_equivalent"] & ~data["cosmic_ray.killed_by_own"]])}"
)
print(
    f"Equivalence-claimed mutants not killed by test from any run: {len(data[data["claimed_equivalent"] & ~data["cosmic_ray.killed_by_any"]])}"
)


# %% Plot percentage of equivalence claims

ticks = []
percentages_killed = []
percentages_unkilled = []

for preset in data["settings.preset_name"].unique():
    if preset == "baseline_without_iterations":
        continue

    for project in data["project"].unique():
        target_data = data[(data["project"] == project) & (data["settings.preset_name"] == preset)]
        claims_unkilled = target_data[(target_data["outcome"] == EQUIVALENT) & ~target_data["cosmic_ray.killed_by_any"]]
        claims_killed = target_data[(target_data["outcome"] == EQUIVALENT) & target_data["cosmic_ray.killed_by_any"]]
        ticks.append(f"{project} {preset}")
        percentages_killed.append(len(claims_killed) / len(target_data))
        percentages_unkilled.append(len(claims_unkilled) / len(target_data))

fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
x = np.arange(len(ticks))
bar1 = ax.bar(x, percentages_unkilled, color=cmap_colors[0])
bar2 = ax.bar(
    x,
    percentages_killed,
    color=cmap_colors[1],
    bottom=percentages_unkilled,
)
ax.set_xticks(x)
ax.set_xticklabels(ticks, rotation=90)
ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.2f}%")
ax.legend(
    [bar1, bar2],
    ["Mutants claimed equivalent (not killed by any test)", "Mutants claimed equivalent (killed by some test)"],
)
plt.show()


# %% Plot percentage of failed runs

ticks = []
percentages_killed = []
percentages_unkilled = []

for preset in data["settings.preset_name"].unique():
    if preset == "baseline_without_iterations":
        continue

    for project in data["project"].unique():
        target_data = data[(data["project"] == project) & (data["settings.preset_name"] == preset)]
        failed_runs_unkilled = target_data[(target_data["outcome"] == FAIL) & ~target_data["cosmic_ray.killed_by_any"]]
        failed_runs_killed = target_data[(target_data["outcome"] == FAIL) & target_data["cosmic_ray.killed_by_any"]]
        ticks.append(f"{project} {preset}")
        percentages_killed.append(len(failed_runs_killed) / len(target_data))
        percentages_unkilled.append(len(failed_runs_unkilled) / len(target_data))

fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
x = np.arange(len(ticks))
bar1 = ax.bar(x, percentages_unkilled, color=cmap_colors[0])
bar2 = ax.bar(
    x,
    percentages_killed,
    color=cmap_colors[1],
    bottom=percentages_unkilled,
)
ax.set_xticks(x)
ax.set_xticklabels(ticks, rotation=90)
ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.2f}%")
ax.legend(
    [bar1, bar2],
    ["Failed runs (mutant not killed by any test)", "Failed runs (mutant killed by some test)"],
)
plt.show()


# %% Check if any mutant was claimed eqivalent by multiple runs


# %% Sample equivalent and failed runs

column_mapping = {
    "preset": "preset",
    "project": "project",
    "problem.target_path": "target_path",
    "problem.mutant_op": "mutant_op",
    "problem.occurrence": "occurrence",
    "id": "id",
}

claims_count = {}
failed_count = {}
for preset in data["preset"].unique():
    for project in data["project"].unique():
        target_data = data[(data["project"] == project) & (data["preset"] == preset)]
        claims_unkilled = cast(
            pd.DataFrame, target_data[(target_data["outcome"] == EQUIVALENT) & ~target_data["cosmic_ray.killed_by_any"]]
        )
        failed_runs_unkilled = cast(
            pd.DataFrame, target_data[(target_data["outcome"] == FAIL) & ~target_data["cosmic_ray.killed_by_any"]]
        )

        claims_unkilled.rename(columns=column_mapping).assign(equivalent=None, comment=None).sample(
            frac=1, random_state=1
        ).to_csv(
            OUTPUT_PATH / f"equivalence_claims_of_unkilled_mutants_{preset}_{project}.csv",
            columns=[*column_mapping.values(), "equivalent", "comment"],
        )
        failed_runs_unkilled.rename(columns=column_mapping).assign(equivalent=None, comment=None).sample(
            frac=1, random_state=1
        ).to_csv(
            OUTPUT_PATH / f"failed_runs_with_unkilled_mutants_{preset}_{project}.csv",
            columns=[*column_mapping.values(), "equivalent", "comment"],
        )

        for mutant_id in claims_unkilled["mutant_id"][:20]:
            count = claims_count.get(mutant_id, 0)
            claims_count[mutant_id] = count + 1
        for mutant_id in failed_runs_unkilled["mutant_id"][:20]:
            count = failed_count.get(mutant_id, 0)
            failed_count[mutant_id] = count + 1


# %%

print(len(data[(data["preset"] == "debugging_one_shot") & (data["mutant.num_equivalence_claims"] == 0)]))
print(len(data[(data["preset"] == "debugging_one_shot") & (data["mutant.num_equivalence_claims"] == 1)]))
print(len(data[(data["preset"] == "debugging_one_shot") & (data["mutant.num_equivalence_claims"] == 2)]))
print(len(data[(data["preset"] == "debugging_one_shot") & (data["mutant.num_equivalence_claims"] == 3)]))
print(len(data[(data["preset"] == "debugging_one_shot") & (data["mutant.num_equivalence_claims"] == 4)]))
print(len(data[data["claimed_equivalent"]]))

# %%

print(len(data[(data["preset"] == "debugging_one_shot")]))
print(len(data[(data["preset"] == "debugging_zero_shot")]))
print(len(data[(data["preset"] == "baseline_with_iterations")]))
print(len(data[(data["preset"] == "baseline_without_iterations")]))
