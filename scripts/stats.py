# %% Imports and constants

import json
import os
import sqlite3
import math
import re
from functools import partial, reduce
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, NamedTuple, Tuple, cast

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm
from tqdm import tqdm

pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: f"{int(x):d}" if math.floor(x) == x else f"{x:.3f}")
# pd.DataFrame.__str__ = pd.DataFrame.to_string
# pd.DataFrame.__repr__ = pd.DataFrame.to_string

JsonObj = Dict[str, Any]
OUTPUT_PATH = Path("/tmp/out")
OUTPUT_PATH.mkdir(exist_ok=True)


# %% Find result directory and mutants

if "get_ipython" in locals():
    print("Running as ipython kernel")
    REPO_PATH = Path(os.getcwd()).parent
else:
    REPO_PATH = Path(__file__).parent
    print("Running as script")

RESULTS_DIR = REPO_PATH / "guut_emse_results"
PYNGUIN_TESTS_DIR = REPO_PATH / "pynguin_emse_tests"
MUTANTS_DIR = REPO_PATH / "emse_projects_data" / "mutants_sampled"


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
                exec_result["output"] = exec_result["output"][:5000]

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
                exec_result["output"] = exec_result["output"][:5000]

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


class CosmicRayInfo(NamedTuple):
    killed: bool
    output: str


FAILED_TEST_REGEX = re.compile(r"TEST FAILED: \[([^\]]+)\]")


def test_outcome_kills(cosmic_ray_outcome):
    if cosmic_ray_outcome not in ["SURVIVED", "KILLED"]:
        raise Exception(f"Unexpected test outcome: {cosmic_ray_outcome}")
    return cosmic_ray_outcome == "KILLED"


def get_failing_tests_for_output(output: str) -> List[str]:
    return [match.group(1) for match in re.finditer(FAILED_TEST_REGEX, output)]


def read_mutant_results(mutants_file: Path, project: str) -> Dict[MutantId, CosmicRayInfo]:
    con = sqlite3.connect(mutants_file)
    cur = con.execute("""
        select
            module_path,
            operator_name,
            occurrence,
            test_outcome,
            output
        from mutation_specs
        left join work_results
            on mutation_specs.job_id = work_results.job_id;
    """)
    mutants = cur.fetchall()
    con.close()
    return {MutantId(project, m[0], m[1], m[2]): CosmicRayInfo(test_outcome_kills(m[3]), m[4]) for m in mutants}


cosmic_ray_results = {}
# Maps (project, preset) to a map that maps (target_path, mutant_op, occurrence) to mutant test result
for mutants_path in RESULTS_DIR.glob("*/cosmic-ray*/mutants.sqlite"):
    results_dir = (mutants_path / ".." / "..").resolve()
    parts = results_dir.name.split("_")
    preset = "_".join(parts[:3])
    package = "_".join(parts[3:-1])
    project = package_to_project[package]
    cosmic_ray_results[(project, preset, "exitfirst")] = read_mutant_results(
        results_dir / "cosmic-ray" / "mutants.sqlite", project
    )
    cosmic_ray_results[(project, preset, "full")] = read_mutant_results(
        results_dir / "cosmic-ray-full" / "mutants.sqlite", project
    )


data["cosmic_ray.killed_by_own"] = data.apply(
    lambda row: cosmic_ray_results[(row["project"], row["preset"], "exitfirst")][mutant_id_from_row(row)].killed,
    axis=1,
)
presets = list(data["preset"].unique())
data["cosmic_ray.killed_by_any"] = data.apply(
    lambda row: any(
        cosmic_ray_results[(row["project"], preset, "exitfirst")][mutant_id_from_row(row)].killed for preset in presets
    ),
    axis=1,
)

data["cosmic_ray_full.killed_by_own"] = data.apply(
    lambda row: cosmic_ray_results[(row["project"], row["preset"], "full")][mutant_id_from_row(row)].killed,
    axis=1,
)
presets = list(data["preset"].unique())
data["cosmic_ray_full.killed_by_any"] = data.apply(
    lambda row: any(
        cosmic_ray_results[(row["project"], preset, "full")][mutant_id_from_row(row)].killed for preset in presets
    ),
    axis=1,
)
data["cosmic_ray_full.failing_tests"] = data.apply(
    lambda row: get_failing_tests_for_output(
        cosmic_ray_results[(row["project"], row["preset"], "full")][mutant_id_from_row(row)].output
    ),
    axis=1,
)

del cosmic_ray_results


# %% Read Pynguin cosmic-ray results

PYNGUIN_FAILED_TEST_REGEX = re.compile(r"Test failed: (.*)")


def get_failing_tests_for_pynguin_output(output: str) -> List[str]:
    return [match.group(1) for match in re.finditer(PYNGUIN_FAILED_TEST_REGEX, output)]


pynguin_data = dict(project=[], index=[], mutant_id=[], killed=[], failing_tests=[])
# Maps (project, preset) to a map that maps (target_path, mutant_op, occurrence) to mutant test result
for mutants_path in PYNGUIN_TESTS_DIR.glob("*/*/cosmic-ray/mutants.sqlite"):
    project = mutants_path.parent.parent.parent.name
    index = mutants_path.parent.parent.name
    if (project, index) in [("flutils", "06"), ("flutils", "09"), ("flake8", "04")]:
        "skip for now, due to errors"
        continue
    results = read_mutant_results(mutants_path, project)
    for mutant_id, cosmic_ray_info in results.items():
        pynguin_data["project"].append(project)
        pynguin_data["index"].append(int(index))
        pynguin_data["mutant_id"].append(mutant_id)
        pynguin_data["killed"].append(cosmic_ray_info.killed)
        pynguin_data["failing_tests"].append(get_failing_tests_for_pynguin_output(cosmic_ray_info.output))
pynguin_data = pd.DataFrame(pynguin_data)


# %% Sum up Pynguin results

Project = str


class PynguinCosmicRayRunSummary(NamedTuple):
    index_: int
    kills: int


best_pynguin_runs: Dict[Project, PynguinCosmicRayRunSummary] = {}
aggregated_pynguin_data = dict(project=[], index=[], killed=[])
for project in pynguin_data["project"].unique():
    for index in range(30):
        index += 1
        if (project, index) in [("flutils", "06"), ("flutils", "09"), ("flake8", "04")]:
            "skip for now, due to errors"
            continue
        target_data = (
            pynguin_data
            # filter by project
            .loc[lambda df: df["project"] == project]
            # filter by index
            .loc[lambda df: df["index"] == index]
        )

        num_kills = target_data["killed"].sum()
        aggregated_pynguin_data["project"].append(project)
        aggregated_pynguin_data["index"].append(index)
        aggregated_pynguin_data["killed"].append(num_kills)
        max_index, max_kills = best_pynguin_runs.get(project, (-1, -1))
        if num_kills > max_kills:
            best_pynguin_runs[project] = PynguinCosmicRayRunSummary(index, num_kills)
aggregated_pynguin_data = pd.DataFrame(aggregated_pynguin_data)


# %% Check which mutants are killed by pynguin runs


class PynguinCosmicRayMutantStatus(NamedTuple):
    killed_by_best: bool
    killed_by_any: bool


pynguin_mutant_statuses: Dict[MutantId, PynguinCosmicRayMutantStatus] = {}
for index, row in pynguin_data.iterrows():
    project = cast(str, row["project"])
    index = row["index"]
    mutant_id = cast(MutantId, row["mutant_id"])
    killed = cast(bool, row["killed"])

    is_best_run = best_pynguin_runs[project].index_ == index

    current_status = pynguin_mutant_statuses.get(mutant_id, PynguinCosmicRayMutantStatus(False, False))
    killed_by_best = current_status.killed_by_best or (is_best_run and killed)
    killed_by_any = current_status.killed_by_any or killed

    pynguin_mutant_statuses[mutant_id] = PynguinCosmicRayMutantStatus(killed_by_best, killed_by_any)

data["pynguin.cosmic_ray.killed_by_best"] = data.apply(
    lambda row: pynguin_mutant_statuses.get(
        row["mutant_id"], PynguinCosmicRayMutantStatus(False, False)
    ).killed_by_best,
    axis=1,
)
data["pynguin.cosmic_ray.killed_by_any"] = data.apply(
    lambda row: pynguin_mutant_statuses.get(row["mutant_id"], PynguinCosmicRayMutantStatus(False, False)).killed_by_any,
    axis=1,
)


# %% Check which mutants are claimed equivalent by multiple runs and what runs they are claimed by


class EquivalenceClaim(NamedTuple):
    preset: str
    id: str


equivalence_claims = {}
for index, row in data.iterrows():
    mutant_id = mutant_id_from_row(row)
    equivalence_claims[mutant_id] = equivalence_claims.get(mutant_id, []) + (
        [EquivalenceClaim(str(row["preset"]), str(row["id"]))] if bool(row["claimed_equivalent"]) else []
    )

presets = list(data["preset"].unique())
data["mutant.equivalence_claims"] = data.apply(
    lambda row: equivalence_claims[mutant_id_from_row(row)],
    axis=1,
)
data["mutant.num_equivalence_claims"] = data["mutant.equivalence_claims"].map(len)
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


# %% Print memory usage

with open("/proc/self/status") as f:
    memusage = f.read().split("VmRSS:")[1].split("\n")[0][:-3]
    print(f"Memory used: {int(memusage.strip()) / (1024**2):.3f} GB")

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

del ticks
del percentages_killed
del percentages_unkilled


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

del ticks
del percentages_killed
del percentages_unkilled


# %% Plot percentage of direct kills

ticks = []
percentages = []  # [direct kill, kill with tests from same run, kill with any tests]

for preset in data["settings.preset_name"].unique():
    for project in data["project"].unique():
        target_data = data[(data["project"] == project) & (data["settings.preset_name"] == preset)]
        ticks.append(f"{project} {preset}")
        percentages.append(
            [
                len(target_data[target_data["mutant_killed"]]) / len(target_data),
                len(target_data[target_data["cosmic_ray.killed_by_own"]]) / len(target_data),
                len(target_data[target_data["cosmic_ray.killed_by_any"]]) / len(target_data),
            ]
        )


color1 = cmap_colors[0]
color3 = cmap_colors[1]
color2 = ((color1[0] + color3[0]) / 2, (color1[1] + color3[1]) / 2, (color1[2] + color3[2]) / 2)

fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
x = np.arange(len(ticks))
bar1 = ax.bar(x, [p[0] for p in percentages], color=cmap_colors[0])
bar2 = ax.bar(
    x,
    [p[1] - p[0] for p in percentages],
    color=cmap_colors[2],
    bottom=[p[0] for p in percentages],
)
bar3 = ax.bar(
    x,
    [p[2] - p[1] for p in percentages],
    color=cmap_colors[4],
    bottom=[p[1] for p in percentages],
)
ax.set_xticks(x)
ax.set_xticklabels(ticks, rotation=90)
ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.2f}%")
ax.legend(
    [bar3, bar2, bar1],
    [
        "Mutants killed by any test.",
        "Mutants killed by tests from the same preset.",
        "Mutants killed by sucessful runs.",
    ],
)
plt.show()

del ticks
del percentages


# %% Number of mutants with 0/1/2/3 claims

print(f"mutants with {0} claims: {len(data[data["mutant.num_equivalence_claims"] == 0])//4:4d}")
print(f"mutants with {1} claims: {len(data[data["mutant.num_equivalence_claims"] == 1])//4:4d}")
print(f"mutants with {2} claims: {len(data[data["mutant.num_equivalence_claims"] == 2])//4:4d}")
print(f"mutants with {3} claims: {len(data[data["mutant.num_equivalence_claims"] == 3])//4:4d}")
print(f"mutants with {4} claims: {len(data[data["mutant.num_equivalence_claims"] == 4])//4:4d}")
print(f"total number of claims: {len(data[data["claimed_equivalent"]])}")


# %% Number of unkilled mutants with 1/2/3 claims per preset and project

for preset in data["preset"].unique():
    if preset == "baseline_without_iterations":
        continue
    for project in data["project"].unique():
        target_data = data[(data["project"] == project) & (data["preset"] == preset)]
        claims_unkilled = cast(
            pd.DataFrame, target_data[(target_data["outcome"] == EQUIVALENT) & ~target_data["cosmic_ray.killed_by_any"]]
        )
        killed_by_1 = len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 1])
        killed_by_2 = len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 2])
        killed_by_3 = len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 3])
        print(f"{preset:<25} {project:<20} {killed_by_1:3} {killed_by_2:3} {killed_by_3:3}")

# %% Number of unkilled mutants with 1/2/3 claims per project

for project in data["project"].unique():
    target_data = data[(data["project"] == project)]
    claims_unkilled = cast(
        pd.DataFrame, target_data[(target_data["outcome"] == EQUIVALENT) & ~target_data["cosmic_ray.killed_by_any"]]
    )
    print(
        f"{project:<20} {len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 1]):3} {len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 2])//4:3} {len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 3])//4:3}"
    )


# %% Sample equivalent and failed runs

# include all of baseline (only 34 unkilled claimed mutants)
sampled_mutants = data[
    ((data["preset"] == "baseline_with_iterations") | (data["preset"] == "baseline_without_iterations"))
    & (data["outcome"] == EQUIVALENT)
    & (~data["cosmic_ray.killed_by_any"])
]

for project in data["project"].unique():
    # The "cosmic_ray.killed_by_any" and "mutant.num_equivalence_claims" columns are replicated for each preset.
    # Therefore, we only look at the "debugging_one_shot" data here.
    target_data = data[(data["project"] == project) & (data["preset"] == "debugging_one_shot")]

    # include 5 mutants for each number of claims
    for num_claims in [3, 2, 1, 0]:
        new_mutants = cast(
            pd.DataFrame,
            target_data[
                (target_data["mutant.num_equivalence_claims"] == num_claims)
                & (~target_data["cosmic_ray.killed_by_any"])
                & (
                    ~cast(pd.Series, target_data["mutant_id"]).isin(
                        cast(pd.Series, sampled_mutants["mutant_id"]).unique()
                    )
                )
            ],
        )
        new_sample = new_mutants.sample(n=min(len(new_mutants), 5), random_state=1)
        # print(f"sampled {len(new_sample)} with {num_claims} claims from {project}")
        sampled_mutants = pd.concat([sampled_mutants, new_sample])

# Double-check that we don't have any duplicate mutants
assert len(sampled_mutants) == len(set(sampled_mutants["mutant_id"]))

print(f"{len(sampled_mutants):4d} mutants in total")
print()

for num_claims in [0, 1, 2, 3]:
    print(
        f"{len(sampled_mutants[sampled_mutants["mutant.num_equivalence_claims"] == num_claims]):4d} mutants with {num_claims} claims"
    )
print()

for project in data["project"].unique():
    print(f"{len(sampled_mutants[sampled_mutants["project"] == project]):4d} mutants from {project}")
print()

sampled_mutants_per_preset = {}
for index, row in sampled_mutants.iterrows():
    for claim in row["mutant.equivalence_claims"]:
        sampled_mutants_per_preset[claim.preset] = sampled_mutants_per_preset.get(claim.preset, []) + [claim.preset]
for preset in data["preset"].unique():
    print(f"{len(sampled_mutants_per_preset.get(preset, [])):4d} mutants from {preset}")
print()
del sampled_mutants_per_preset

# Write to CSV

column_mapping = {
    "project": "project",
    "problem.target_path": "target_path",
    "problem.mutant_op": "mutant_op",
    "problem.occurrence": "occurrence",
    "claim1": "claim1",
    "claim2": "claim2",
    "claim3": "claim3",
}

sampled_mutants["claim1"] = cast(pd.Series, sampled_mutants["mutant.equivalence_claims"]).map(
    lambda l: l[0].id if len(l) > 0 else None
)
sampled_mutants["claim2"] = cast(pd.Series, sampled_mutants["mutant.equivalence_claims"]).map(
    lambda l: l[1].id if len(l) > 1 else None
)
sampled_mutants["claim3"] = cast(pd.Series, sampled_mutants["mutant.equivalence_claims"]).map(
    lambda l: l[2].id if len(l) > 2 else None
)

sampled_mutants = cast(pd.DataFrame, sampled_mutants).sort_values(
    by=["project", "problem.target_path", "problem.mutant_op", "problem.occurrence"]
)

sampled_mutants.rename(columns=column_mapping).assign(equivalent=None, unsure=None, comment=None).to_csv(
    OUTPUT_PATH / "sampled_mutants.csv",
    columns=[*column_mapping.values(), "equivalent", "unsure", "comment"],
)


# %% Read sample data

sample_data = pd.read_csv(REPO_PATH / "samples" / "sampled_mutants.csv")
column_mapping = {
    "equivalent": "sample.equivalent",
    "unsure": "sample.unsure",
    "change": "sample.change",
    "kill_method": "sample.kill_method",
    "comment": "sample.comment",
}


def sample_row_to_dict(row):
    return {val: row[key] for key, val in column_mapping.items()}


def sample_row_to_mutant_id(row):
    return MutantId(
        project=row["project"],
        target_path=row["target_path"],
        mutant_op=row["mutant_op"],
        occurrence=row["occurrence"],
    )


def convert_csv_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if np.isnan(value):
        return None
    if value == 0:
        return False
    if value == 1:
        return True
    return value


sample_data = {sample_row_to_mutant_id(row): sample_row_to_dict(row) for _, row in sample_data.iterrows()}

for name in column_mapping.values():
    data[name] = data["mutant_id"].map(
        lambda mutant_id: convert_csv_value(sample_data[mutant_id][name]) if mutant_id in sample_data else None
    )
    # data[name] = data[name].map(lambda val: None if np.isnan(val) else None)
data["sampled"] = data["sample.equivalent"].map(lambda b: b is not None)


# %% Check which percentage of mutants with n claims were (not) killed

mutant_data = data[data["preset"] == "debugging_one_shot"]

for num_claims in [0, 1, 2, 3]:
    percentage = len(
        data[(data["mutant.num_equivalence_claims"] == num_claims) & (~data["cosmic_ray.killed_by_any"])]
    ) / len(data[data["mutant.num_equivalence_claims"] == num_claims])
    print(f"{percentage * 100:5.2f}% of mutants with {num_claims} claims weren't killed by any test")

del mutant_data


# %%

data[data["cosmic_ray.killed_by_own"] != data["cosmic_ray_full.killed_by_own"]].apply(
    lambda row: (row["preset"], row["mutant_id"].project), axis=1
).unique()
# data[["mutant_id"]][data["cosmic_ray.killed_by_own"] != data["cosmic_ray_full.killed_by_own"]]
# (data["cosmic_ray.killed_by_own"] != data["cosmic_ray_full.killed_by_own"]).value_counts()

# %%

print(len(data[data["cosmic_ray.killed_by_own"]]))
print(len(data[data["pynguin.cosmic_ray.killed_by_any"]]))


# %% Overview over sampled mutants

target_data = data[data["preset"] == "debugging_one_shot"]
for n in target_data["mutant.num_equivalence_claims"].unique():
    print(
        f"claims: {n}, equivalent: yes -> {len(target_data[target_data["sampled"] & (target_data["sample.equivalent"] == True) & (target_data["mutant.num_equivalence_claims"] == n)])}"
    )
    print(
        f"claims: {n}, equivalent: no -> {len(target_data[target_data["sampled"] & (target_data["sample.equivalent"] == False) & (target_data["mutant.num_equivalence_claims"] == n)])}"
    )
    print(
        f"claims: {n}, equivalent: yes, unsure -> {len(target_data[target_data["sampled"] & target_data["sample.unsure"] & (target_data["sample.equivalent"] == True) & (target_data["mutant.num_equivalence_claims"] == n)])}"
    )
    print(
        f"claims: {n}, equivalent: no, unsure -> {len(target_data[target_data["sampled"] & target_data["sample.unsure"] & (target_data["sample.equivalent"] == False) & (target_data["mutant.num_equivalence_claims"] == n)])}"
    )
