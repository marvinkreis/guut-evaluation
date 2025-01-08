# %% Docs


# Dataframes:
#
# - data
#     - Contains the results from running guut on 10 EMSE projects.
#     - Each row contains the result of one "loop" on one mutant.
#     - Indexed by ("preset", "mutant_id") = ("preset", "project", "target_path", "mutant_op", "occurrence")
#     - Available columns: search for "data_columns"
#
# - pynguin_data
#     - Contains the cosmic-ray results from running the Pynguin-generated tests on 10 EMSE projects.
#     - Each row contains the result of running cosmic-ray with the generated test suite on one mutant.
#     - Indexed by ("index", "mutant_id") = ("index", "project", "target_path", "mutant_op", "occurrence")
#     - Available columns: TODO
#
# - aggregated_pynguin_data
#     - TODO: can this be replaced with a simple groupby?
#
# - sample_data
#
#
# Maps:
#
# - coverage_map
#     - Contains the coverage of the LLM-generated test suites each EMSE project.
#     - Each entry contains the coverage for an entire test suite for one project.
#     - Indexed by ("preset", "project").
#
# - pynguin_coverage_map
#     - Contains the coverage of the Pynguin-generated test suites each EMSE project.
#     - Each entry contains the coverage for an entire test suite for one project.
#     - Indexed by ("project", "index") where 1 <= index <= 30.
#
# - all_seen_lines_map / all_seen_branches_map
#     - Contains all (executed or missing) lines/branches from coverage.py for each EMSE project.
#     - Each entry contains a set of lines of one project.
#     - Indexed by "project"


# %% Imports and constants
# ======================================================================================================================

import json
import os
import sqlite3
import math
import re
from functools import partial, reduce
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, NamedTuple, Tuple, cast, Callable
from scipy.stats import mannwhitneyu

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm
from matplotlib.figure import Figure
from tqdm import tqdm

pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option(
    "display.float_format", lambda x: f"{int(x):d}" if math.floor(x) == x else f"{x:.3f}" if x > 1000 else f"{x:.6f}"
)

type JsonObj = Dict[str, Any]
OUTPUT_PATH = Path("/tmp/out")
OUTPUT_PATH.mkdir(exist_ok=True)


# %% Find result directories


if "get_ipython" in locals():
    print("Running as ipython kernel")
    REPO_PATH = Path(os.getcwd()).parent
else:
    REPO_PATH = Path(__file__).parent
    print("Running as script")

RESULTS_DIR = REPO_PATH / "guut_emse_results"
PYNGUIN_TESTS_DIR = REPO_PATH / "pynguin_emse_tests"
MUTANTS_DIR = REPO_PATH / "emse_projects_data" / "mutants_sampled"


# %% Define some constants


PRESETS = [
    "baseline_without_iterations",
    "baseline_with_iterations",
    "debugging_zero_shot",
    "debugging_one_shot",
]

PRESET_NAMES = [
    "Baseline",
    "Iterative",
    "Scientific (0-shot)",
    "Scientific (1-shot)",
]

PROJECTS = [
    "apimd",
    "codetiming",
    "dataclasses-json",
    "docstring_parser",
    "flake8",
    "flutes",
    "flutils",
    "httpie",
    "pdir2",
    "python-string-utils",
]

PACKAGE_TO_PROJECT = {
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

SUCCESS = "success"
EQUIVALENT = "equivalent"
FAIL = "fail"

OUTCOMES = [SUCCESS, EQUIVALENT, FAIL]
OUTCOME_NAMES = ["Success", "Claimed Equivalent", "Failed"]

AGGS = ["mean", "median", "min", "max", "sum"]


# %% Helpers
# ======================================================================================================================


# %% Decorator to separate code into blocks (from stackoverflow.com/a/7718776)


def block(function):
    return function()


# %% Decorator to save plots


def savefig(function):
    def outer_function():
        retval = function()

        if isinstance(retval, Figure):
            fig = retval
            fig.savefig(OUTPUT_PATH / f"{function.__name__}.png")
            fig.savefig(OUTPUT_PATH / f"{function.__name__}.pdf")

        if isinstance(retval, list):
            for index, entry in enumerate(retval):
                fig = entry
                fig.savefig(OUTPUT_PATH / f"{function.__name__}_{index+1:02}.png")
                fig.savefig(OUTPUT_PATH / f"{function.__name__}.{index+1:02}.pdf")

    return outer_function


# %% Excluded Pynguin runs


def is_pynguin_run_excluded(project, index):
    index = int(index)
    if project == "pdir2":  # Breaks Pynguin, because pdir2 replaces its own module with a function.
        return True
    if (project, index) in [
        ("flutils", 6),  # Tests delete the package under test.
        ("flutils", 9),  # Tests delete the package under test.
        ("flutils", 10),  # Breaks coverage.py.
        ("flutils", 20),  # Breaks coverage.py.
        ("flake8", 4),  # Tests cause pytest to raise an OSError. Most mutants are killed even though the tests pass.
        ("apimd", 24),  # Missing in the results.
    ]:
        return True
    return False


def is_pynguin_project_excluded(project):
    return project == "pdir2"


def get_num_pynguin_runs_per_project(project: str):
    if project == "flutils":
        return 30 - 4
    elif project == "flake8":
        return 30 - 1
    elif project == "apimd":
        return 30 - 1
    else:
        return 30


# %% Parse info from the results directory name


class LongId(NamedTuple):
    preset: str
    package: str
    project: str
    id: str

    @staticmethod
    def parse(long_id: str) -> "LongId":
        parts = long_id.split("_")
        preset = "_".join(parts[:3])
        package = "_".join(parts[3:-1])
        project = PACKAGE_TO_PROJECT[package]
        id = parts[-1]
        return LongId(preset, package, project, id)


# %% Format floats


def format_perecent(digits=0):
    return lambda num, pos: f"{num * 100:.{digits}f}%"


# %% Read and prepare data
# ======================================================================================================================


# %% Load json result files


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
                    # TODO: read coverage for all files here, not just the target path
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
                    # TODO: read coverage for all files here, not just the target path
                    if raw_file_coverage := coverage["raw"]["files"][f"{module_name}/{target_path}"]:
                        coverage["covered_branches"] = raw_file_coverage["executed_branches"]
                        coverage["missed_branches"] = raw_file_coverage["missing_branches"]
                    else:
                        coverage["covered_branches"] = []
                        coverage["missed_branches"] = []
                    del coverage["raw"]
                exec_result["output"] = exec_result["output"][:5000]


def read_single_result(path: Path) -> JsonObj:
    with path.open("r") as f:
        result = json.load(f)
    prepare_loop_result(result)
    result["path"] = str(path)
    return result


def read_data():
    results_paths = list(RESULTS_DIR.glob("*/loops/*/result.json"))

    with Pool(8) as pool:
        results_json = list(
            tqdm(
                pool.imap_unordered(read_single_result, results_paths, chunksize=100),
                total=len(results_paths),
                desc="Loading tons of json files",
            )
        )

    data = pd.json_normalize(results_json)
    data = data.sort_values(by=["long_id", "problem.target_path", "problem.mutant_op", "problem.occurrence"])
    return data


if "data" not in locals():
    data = read_data()
else:
    raise Exception("data is already in memory. Refusing to read the files again.")


# %% Add project name to data


data["project"] = data["problem.module_path"].map(lambda path: PACKAGE_TO_PROJECT[path.rsplit("/")[-1]])


# %% Add mutant identifier to data


class MutantId(NamedTuple):
    project: str
    target_path: str
    mutant_op: str
    occurrence: int

    @staticmethod
    def from_row(row):
        return MutantId(
            project=row["project"],
            target_path=row["problem.target_path"],
            mutant_op=row["problem.mutant_op"],
            occurrence=row["problem.occurrence"],
        )


data["mutant_id"] = data.apply(MutantId.from_row, axis=1)


# %% Compute outcome for simplicity


# TODO: There are also some runs that claimed the mutant is equivalent and then killed the mutant.
# These either need to be handled differently of this needs to be explained in the paper/thesis.


@block
def add_outcome_to_data():
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


@block
def add_mutant_lines_to_data():
    mutant_lines = {}
    for project in data["project"].unique():
        mutant_lines.update(read_mutant_lines(MUTANTS_DIR / f"{project}.sqlite", project))

    data["mutant_lines"] = data["mutant_id"].map(mutant_lines.get)
    data["mutant_lines.start"] = data["mutant_lines"].map(lambda t: t[0])
    data["mutant_lines.end"] = data["mutant_lines"].map(lambda t: t[1])


# %% Read combined coverage info (coverage of all (non-flaky) tests from a project)


# coverage.py only counts lines or branches as missing if the containing module was at least loaded.
# This leads to a different number of total branches between the measurements. The coverage measures can however
# still easily be compared by dividing the number of hit lines/branches by the number of the union of all recorded
# lines/branches. This means that the coverage might still be off compared to the "true" coverage containing all lines # and branches in the project, but all coverage measurements will be equally off.


class LineId(NamedTuple):
    file_name: str
    line: int


class BranchId(NamedTuple):
    file_name: str
    line_1: int
    line_2: int


class Coverage(NamedTuple):
    missing_lines: List[LineId]
    executed_lines: List[LineId]
    missing_branches: List[BranchId]
    executed_branches: List[BranchId]


def read_coverage_py_json(coverage_json: JsonObj) -> Coverage:
    missing_lines = []
    executed_lines = []
    missing_branches = []
    executed_branches = []
    for file_name, file in coverage_json["files"].items():
        missing_lines += [LineId(file_name, line) for line in file["missing_lines"]]
        executed_lines += [LineId(file_name, line) for line in file["executed_lines"]]
        missing_branches += [BranchId(file_name, branch[0], branch[1]) for branch in file["missing_branches"]]
        executed_branches += [BranchId(file_name, branch[0], branch[1]) for branch in file["executed_branches"]]
    return Coverage(missing_lines, executed_lines, missing_branches, executed_branches)


def read_coverage():
    map = {}
    for coverage_path in RESULTS_DIR.glob("*/coverage/coverage.json"):
        results_dir = (coverage_path / ".." / "..").resolve()
        id = LongId.parse(results_dir.name)
        with coverage_path.open("r") as f:
            coverage_json = json.load(f)
        map[(id.preset, id.project)] = read_coverage_py_json(coverage_json)
    return map


coverage_map: Dict[Tuple[str, str], Coverage] = read_coverage()


def read_pynguin_coverage():
    map = {}
    for coverage_path in PYNGUIN_TESTS_DIR.glob("*/*/coverage/coverage.json"):
        project = coverage_path.parent.parent.parent.name
        index = coverage_path.parent.parent.name
        if is_pynguin_run_excluded(project, index):
            continue
        with coverage_path.open("r") as f:
            coverage_json = json.load(f)
        map[(project, int(index))] = read_coverage_py_json(coverage_json)
    return map


pynguin_coverage_map: Dict[Tuple[str, int], Coverage] = read_pynguin_coverage()


def sum_code_elements():
    lines_map = {}
    branches_map = {}

    for project in data["project"].unique():
        lines = set()
        branches = set()

        for preset in PRESETS:
            lines.update(coverage_map[(preset, project)].executed_lines)
            lines.update(coverage_map[(preset, project)].missing_lines)
            branches.update(coverage_map[(preset, project)].executed_branches)
            branches.update(coverage_map[(preset, project)].missing_branches)

        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue
            lines.update(pynguin_coverage_map[(project, index)].executed_lines)
            lines.update(pynguin_coverage_map[(project, index)].missing_lines)
            branches.update(pynguin_coverage_map[(project, index)].executed_branches)
            branches.update(pynguin_coverage_map[(project, index)].missing_branches)

        lines_map[project] = lines
        branches_map[project] = branches

    return lines_map, branches_map


all_seen_lines_map, all_seen_branches_map = sum_code_elements()


# %% Print number of seen lines and branches


@block
def print_seen_lines_and_branches():
    for project in data["project"].unique():
        lines = [
            len(coverage_map[(preset, project)].missing_lines) + len(coverage_map[(preset, project)].executed_lines)
            for preset in PRESETS
        ]
        branches = [
            len(coverage_map[(preset, project)].missing_branches)
            + len(coverage_map[(preset, project)].executed_branches)
            for preset in PRESETS
        ]
        print(f"{project}: {lines}")
        print(f"{project}: {branches}")


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


@block
def add_cosmic_ray_results_to_data():
    cosmic_ray_results = {}

    # Maps (project, preset) to a map that maps (target_path, mutant_op, occurrence) to mutant test result
    for mutants_path in RESULTS_DIR.glob("*/cosmic-ray*/mutants.sqlite"):
        results_dir = (mutants_path / ".." / "..").resolve()
        id = LongId.parse(results_dir.name)
        cosmic_ray_results[(id.project, id.preset, "exitfirst")] = read_mutant_results(
            results_dir / "cosmic-ray" / "mutants.sqlite", id.project
        )
        cosmic_ray_results[(id.project, id.preset, "full")] = read_mutant_results(
            results_dir / "cosmic-ray-full" / "mutants.sqlite", id.project
        )

    data["cosmic_ray.killed_by_own"] = data.apply(
        lambda row: cosmic_ray_results[(row["project"], row["preset"], "exitfirst")][MutantId.from_row(row)].killed,
        axis=1,
    )
    presets = list(data["preset"].unique())
    data["cosmic_ray.killed_by_any"] = data.apply(
        lambda row: any(
            cosmic_ray_results[(row["project"], preset, "exitfirst")][MutantId.from_row(row)].killed
            for preset in presets
        ),
        axis=1,
    )

    data["cosmic_ray_full.killed_by_own"] = data.apply(
        lambda row: cosmic_ray_results[(row["project"], row["preset"], "full")][MutantId.from_row(row)].killed,
        axis=1,
    )
    presets = list(data["preset"].unique())
    data["cosmic_ray_full.killed_by_any"] = data.apply(
        lambda row: any(
            cosmic_ray_results[(row["project"], preset, "full")][MutantId.from_row(row)].killed for preset in presets
        ),
        axis=1,
    )
    data["cosmic_ray_full.failing_tests"] = data.apply(
        lambda row: get_failing_tests_for_output(
            cosmic_ray_results[(row["project"], row["preset"], "full")][MutantId.from_row(row)].output
        ),
        axis=1,
    )


# %% Read Pynguin cosmic-ray results


PYNGUIN_FAILED_TEST_REGEX = re.compile(r"Test failed: (.*)")


def get_failing_tests_for_pynguin_output(output: str) -> List[str]:
    return [match.group(1) for match in re.finditer(PYNGUIN_FAILED_TEST_REGEX, output)]


pynguin_data = dict(project=[], index=[], mutant_id=[], killed=[], failing_tests=[])
# Maps (project, preset) to a map that maps (target_path, mutant_op, occurrence) to mutant test result
for mutants_path in PYNGUIN_TESTS_DIR.glob("*/*/cosmic-ray/mutants.sqlite"):
    project = mutants_path.parent.parent.parent.name
    index = mutants_path.parent.parent.name
    if is_pynguin_run_excluded(project, index):
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


type Project = str


class PynguinCosmicRayRunSummary(NamedTuple):
    index_: int
    kills: int


best_pynguin_runs: Dict[Project, PynguinCosmicRayRunSummary] = {}
aggregated_pynguin_data = dict(project=[], index=[], killed=[])
for project in pynguin_data["project"].unique():
    for index in range(30):
        index += 1
        if is_pynguin_run_excluded(project, index):
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
    num_kills: int


pynguin_mutant_statuses: Dict[MutantId, PynguinCosmicRayMutantStatus] = {}
for index, row in pynguin_data.iterrows():
    project = cast(str, row["project"])
    index = row["index"]
    mutant_id = cast(MutantId, row["mutant_id"])
    killed = cast(bool, row["killed"])

    is_best_run = best_pynguin_runs[project].index_ == index

    current_status = pynguin_mutant_statuses.get(mutant_id, PynguinCosmicRayMutantStatus(False, False, 0))
    killed_by_best = current_status.killed_by_best or (is_best_run and killed)
    killed_by_any = current_status.killed_by_any or killed
    num_kills = current_status.num_kills + (1 if killed else 0)

    pynguin_mutant_statuses[mutant_id] = PynguinCosmicRayMutantStatus(killed_by_best, killed_by_any, num_kills)

data["pynguin.cosmic_ray.killed_by_best"] = data.apply(
    lambda row: pynguin_mutant_statuses.get(
        row["mutant_id"], PynguinCosmicRayMutantStatus(False, False, 0)
    ).killed_by_best,
    axis=1,
)
data["pynguin.cosmic_ray.killed_by_any"] = data.apply(
    lambda row: pynguin_mutant_statuses.get(
        row["mutant_id"], PynguinCosmicRayMutantStatus(False, False, 0)
    ).killed_by_any,
    axis=1,
)
data["pynguin.cosmic_ray.num_kills"] = data.apply(
    lambda row: pynguin_mutant_statuses.get(row["mutant_id"], PynguinCosmicRayMutantStatus(False, False, 0)).num_kills,
    axis=1,
)


# %% Check which mutants are claimed equivalent by multiple runs and what runs they are claimed by


class EquivalenceClaim(NamedTuple):
    preset: str
    id: str


@block
def add_equivalence_claims():
    equivalence_claims = {}
    for index, row in data.iterrows():
        mutant_id = MutantId.from_row(row)
        equivalence_claims[mutant_id] = equivalence_claims.get(mutant_id, []) + (
            [EquivalenceClaim(str(row["preset"]), str(row["id"]))] if bool(row["claimed_equivalent"]) else []
        )

    data["mutant.equivalence_claims"] = data.apply(
        lambda row: equivalence_claims[MutantId.from_row(row)],
        axis=1,
    )
    data["mutant.num_equivalence_claims"] = data["mutant.equivalence_claims"].map(len)


# %% Helper functions


def get_execution_result(exp_or_test: JsonObj, mutant: bool = False) -> JsonObj | None:
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
        if (exec_result := get_execution_result(exp)) and (coverage := exec_result["coverage"])
    )


data["mutant_covered.by_experiment"] = data.apply(partial(covers_mutant, experiments_or_tests="experiments"), axis=1)
data["mutant_covered.by_test"] = data.apply(partial(covers_mutant, experiments_or_tests="tests"), axis=1)
data["mutant_covered"] = data["mutant_covered.by_experiment"] | data["mutant_covered.by_test"]


# %% Get line coverage from killing tests


def get_line_coverage_from_test(test):
    exec_result = get_execution_result(test)
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
    exec_result = get_execution_result(test)
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


@block
def count_mutants_per_module():
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

    data["module.num_mutants"] = data.apply(
        lambda row: num_mutants[(row["project"], row["problem.target_path"])], axis=1
    )


# %% Get number of coverable lines and branches per file


# TODO: This doesn't include Pynguin's data
# TODO: Is this still relevant?


@block
def combine_seen_lines_and_branches():
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
data["usage.uncached_prompt_tokens"] = data["usage.prompt_tokens"] - data["usage.cached_tokens"]
data["usage.total_tokens"] = data["usage.prompt_tokens"] + data["usage.completion_tokens"]

# prompt: $0.150 / 1M input tokens
data["usage.cost.uncached_prompt_tokens"] = data["usage.uncached_prompt_tokens"] * 0.150 / 1_000_000

# cached prompt: $0.075 / 1M input tokens
data["usage.cost.cached_tokens"] = data["usage.cached_tokens"] * 0.075 / 1_000_000

# completion: $0.600 / 1M input tokens
data["usage.cost.completion_tokens"] = data["usage.completion_tokens"] * 0.600 / 1_000_000

# Add cost in $ for gpt-4o-mini
data["usage.cost"] = (
    data["usage.cost.uncached_prompt_tokens"] + data["usage.cost.cached_tokens"] + data["usage.cost.completion_tokens"]
)


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

# TODO: count equivalence claim as a turn?
data["num_turns"] = data["num_experiments"] + data["num_tests"]


EQUIVALENCE_HEADLINE_REGEX = re.compile(r"^(#+) +([a-zA-Z0-9]+ +)*equiv", re.IGNORECASE)


def count_turns_before_equivalence(conversation):
    count = 0
    for msg in conversation:
        if msg["role"] != "assistant":
            continue

        if any(re.match(EQUIVALENCE_HEADLINE_REGEX, line) for line in msg["content"].splitlines()):
            break
        elif msg["tag"] in ["experiment_stated", "test_stated"]:
            count += 1
        elif msg["tag"] == "claimed_equivalent":
            raise Exception("this shouldn't happen")
    return count


def count_completions_before_equivalence(conversation):
    count = 0
    for msg in conversation:
        if msg["role"] != "assistant":
            continue

        if any(re.match(EQUIVALENCE_HEADLINE_REGEX, line) for line in msg["content"].splitlines()):
            break
        elif msg["tag"] == "claimed_equivalent":
            raise Exception("this shouldn't happen")

        count += 1
    return count


data["num_turns_before_equivalence_claim"] = data["conversation"].map(count_turns_before_equivalence)
data["num_completions_before_equivalence_claim"] = data["conversation"].map(count_completions_before_equivalence)

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
            if (exec_result := get_execution_result(exp)) and "ModuleNotFoundError" in exec_result["output"]
        ]
    )


data["num_experiment_import_errors"] = data["experiments"].map(count_import_errors)
data["num_test_import_errors"] = data["tests"].map(count_import_errors)


# %% Check for differences in exit code and output in all experiments/tests


def has_exit_code_difference(row):
    for exp_or_test in row["experiments"] + row["tests"]:
        if (correct_result := get_execution_result(exp_or_test, mutant=False)) and (
            mutant_result := get_execution_result(exp_or_test, mutant=True)
        ):
            if correct_result["exitcode"] != mutant_result["exitcode"]:
                return True
    return False


def has_output_difference(row):
    for exp_or_test in row["experiments"] + row["tests"]:
        if (correct_result := get_execution_result(exp_or_test, mutant=False)) and (
            mutant_result := get_execution_result(exp_or_test, mutant=True)
        ):
            correct_output = correct_result["output"].replace(correct_result["cwd"], "")
            mutant_output = mutant_result["output"].replace(mutant_result["cwd"], "")
            if correct_output != mutant_output:
                return True
    return False


data["exit_code_diff"] = data.apply(lambda row: has_exit_code_difference(row), axis=1)
data["output_diff"] = data.apply(lambda row: has_output_difference(row), axis=1)


# %% Sample equivalent and failed runs


def sample_mutants():
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
        lambda claims: claims[0].id if len(claims) > 0 else None
    )
    sampled_mutants["claim2"] = cast(pd.Series, sampled_mutants["mutant.equivalence_claims"]).map(
        lambda claims: claims[1].id if len(claims) > 1 else None
    )
    sampled_mutants["claim3"] = cast(pd.Series, sampled_mutants["mutant.equivalence_claims"]).map(
        lambda claims: claims[2].id if len(claims) > 2 else None
    )

    sampled_mutants = cast(pd.DataFrame, sampled_mutants).sort_values(
        by=["project", "problem.target_path", "problem.mutant_op", "problem.occurrence"]
    )

    sampled_mutants.rename(columns=column_mapping).assign(equivalent=None, unsure=None, comment=None).to_csv(
        OUTPUT_PATH / "sampled_mutants.csv",
        columns=[*column_mapping.values(), "equivalent", "unsure", "comment"],
    )

    return sampled_mutants


sampled_mutants = sample_mutants()


# %% Read sample data


@block
def add_sample_results_to_data():
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


# %% Print memory usage


with open("/proc/self/status") as f:
    memusage = f.read().split("VmRSS:")[1].split("\n")[0][:-3]
    print(f"Memory used: {int(memusage.strip()) / (1024**2):.3f} GB")


# %% Print available columns (data_columns)


cols = list(data.columns)
mid1 = int(((len(cols) + 1) // 3))
mid2 = int(((len(cols) + 1) // 3) * 2)
for x, y, z in zip(cols[:mid1] + [""], cols[mid1:mid2] + [""], cols[mid2:] + [""]):
    print(f"{x:<40} {y:<40} {z}")

# %% Misc Plots and Data
# ======================================================================================================================


class ____Misc_Plots_And_Data:  # mark this in the outline
    pass


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


# %% Number of turns


# for preset in data["preset"].unique():
#     target_data = data[data["preset"] == preset]
#
#     fig, ax = plt.subplots(layout="constrained", figsize=(8, 4))
#     bins = np.arange(11) + 1
#
#     ax.set_xticks(bins)
#     ax.hist(
#         target_data["num_turns"],
#         bins=[float(x) for x in bins],
#         rwidth=0.8,
#     )
#     ax.legend(["Number of Conversations with N Turns"], loc="upper left")
#     fig.text(0.00, 1.02, preset)
#     plt.show()


# %% Number of turns per successful / unsuccessful run


# for preset in data["preset"].unique():
#     target_data = data[data["preset"] == preset]
#
#     turns_success = target_data["num_turns"][target_data["outcome"] == SUCCESS]
#     turns_equivalent = target_data["num_turns"][target_data["outcome"] == EQUIVALENT]
#     turns_fail = target_data["num_turns"][target_data["outcome"] == FAIL]
#
#     fig, ax = plt.subplots(layout="constrained", figsize=(8, 4))
#     bins = np.arange(11) + 1
#
#     ax.set_xticks(bins)
#     ax.hist(
#         [turns_success, turns_equivalent, turns_fail],
#         bins=[float(x) for x in bins],
#         rwidth=1,
#     )
#     ax.legend(
#         [
#             "Number of Conversations with N Turns (Success)",
#             "Number of Conversations with N Turns (Claimed Equivalent)",
#             "Number of Conversations with N Turns (Failed)",
#         ],
#         loc="upper right",
#     )
#     fig.text(0.00, 1.02, preset)
#     plt.show()


# %% Plot Coverage


USE_BRANCH_COVERAGE = True


def add_ids(x: List[str], y: List[str]) -> List[str]:
    acc = set(x)
    acc.update(y)
    return list(acc)


add_ids_np = np.frompyfunc(add_ids, 2, 1)

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
            acc_coverage = add_ids_np.accumulate(target_data["coverage.covered_branch_ids"])
            all_coverable_objects = reduce(add_ids, target_data["coverage.all_branch_ids"], [])
        else:
            acc_coverage = add_ids_np.accumulate(target_data["coverage.covered_line_ids"])
            all_coverable_objects = reduce(add_ids, target_data["coverage.all_line_ids"], [])

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
print(f"Mutants killed by tests from the same run: {len(data[data["cosmic_ray_full.killed_by_own"]])}")
print(f"Mutants killed by tests from the any run: {len(data[data["cosmic_ray_full.killed_by_any"]])}")


# %% Equivalent mutants


print(f"Equivalence-claimed mutants: {len(data[data["claimed_equivalent"]])}")
print(
    f"Equivalence-claimed mutants not killed by test from same run: {len(data[data["claimed_equivalent"] & ~data["cosmic_ray_full.killed_by_own"]])}"
)
print(
    f"Equivalence-claimed mutants not killed by test from any run: {len(data[data["claimed_equivalent"] & ~data["cosmic_ray_full.killed_by_any"]])}"
)


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
            pd.DataFrame,
            target_data[(target_data["outcome"] == EQUIVALENT) & ~target_data["cosmic_ray_full.killed_by_any"]],
        )
        killed_by_1 = len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 1])
        killed_by_2 = len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 2])
        killed_by_3 = len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 3])
        print(f"{preset:<25} {project:<20} {killed_by_1:3} {killed_by_2:3} {killed_by_3:3}")


# %% Number of unkilled mutants with 1/2/3 claims per project


for project in data["project"].unique():
    target_data = data[(data["project"] == project)]
    claims_unkilled = cast(
        pd.DataFrame,
        target_data[(target_data["outcome"] == EQUIVALENT) & ~target_data["cosmic_ray_full.killed_by_any"]],
    )
    print(
        f"{project:<20} {len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 1]):3} {len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 2])//4:3} {len(claims_unkilled[claims_unkilled["mutant.num_equivalence_claims"] == 3])//4:3}"
    )


# %% Check which percentage of mutants with n claims were (not) killed


for num_claims in [0, 1, 2, 3]:
    percentage = len(
        data[(data["mutant.num_equivalence_claims"] == num_claims) & (~data["cosmic_ray_full.killed_by_any"])]
    ) / len(data[data["mutant.num_equivalence_claims"] == num_claims])
    print(f"{percentage * 100:5.2f}% of mutants with {num_claims} claims weren't killed by any test")


# %% Anomalies


data[data["cosmic_ray_full.killed_by_own"] != data["cosmic_ray_full.killed_by_own"]].apply(
    lambda row: (row["preset"], row["mutant_id"].project), axis=1
).unique()
# data[["mutant_id"]][data["cosmic_ray_full.killed_by_own"] != data["cosmic_ray_full.killed_by_own"]]
# (data["cosmic_ray_full.killed_by_own"] != data["cosmic_ray_full.killed_by_own"]).value_counts()


# %% Overview over sampled mutants


target_data = data[data["preset"] == "debugging_one_shot"]
for n in target_data["mutant.num_equivalence_claims"].unique():  # pyright: ignore
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


# %% Define plot helper functions
# ======================================================================================================================


class ____Plot_Helpers:  # mark this in the outline
    pass


def query_group(grouped_data, group, fun, default):
    try:
        return fun(grouped_data.get_group(group))
    except KeyError:
        return default


def plot_bar_per_group(
    grouped_data: Any,
    groups: List[Tuple[str, str]],
    cols: List[Tuple[str, Callable[[Any], Any]]],
    customization: Callable[[Any, Any], None] | None = None,
):
    values = [[query_group(grouped_data, group_key, fun, 0) for group_key, group_name in groups] for name, fun in cols]

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    for i in range(len(values)):
        ax.bar(x=np.arange(4), bottom=np.sum(values[:i], axis=0), height=values[i])

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([group_name for group_key, group_name in groups], rotation=90)
    fig.legend([c[0] for c in cols], loc="lower right")

    if customization:
        customization(fig, ax)

    return fig


def plot_box_per_group(
    grouped_data: Any,
    groups: List[Tuple[str, str]],
    col_name: str,
    col_fun: Callable[[Any], Any],
    customization: Callable[[Any, Any], None] | None = None,
):
    values = [query_group(grouped_data, group_key, col_fun, 0) for group_key, group_name in groups]

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.boxplot(values)

    ax.set_xticks(np.arange(len(groups)) + 1)
    ax.set_xticklabels([group_name for group_key, group_name in groups], rotation=90)
    fig.legend([col_name], loc="lower right")

    if customization:
        customization(fig, ax)

    return fig


def plot_violin_per_group(
    grouped_data: Any,
    groups: List[Tuple[str, str]],
    col_name: str,
    col_fun: Callable[[Any], Any],
    customization: Callable[[Any, Any], None] | None = None,
):
    values = [query_group(grouped_data, group_key, col_fun, [0]) for group_key, group_name in groups]

    fig, ax = plt.subplots(layout="constrained", figsize=(6, 6))
    ax.violinplot(values, showmeans=True, widths=np.repeat(1, len(groups)))

    ax.set_xticks(np.arange(len(groups)) + 1)
    ax.set_xticklabels([group_name for group_key, group_name in groups], rotation=90)
    fig.legend([col_name], loc="lower right")

    if customization:
        customization(fig, ax)

    return fig


# %% Shelved Plots and Data
# ======================================================================================================================


class ____Shelved_Plots_And_Data:  # mark this in the outline
    pass


# %% Plot percentage of direct kills


@block
@savefig
def plot_percentage_of_direct_kills():
    ticks = []
    percentages = []  # [direct kill, kill with tests from same run, kill with any tests]

    for preset in data["settings.preset_name"].unique():
        for project in data["project"].unique():
            target_data = data[(data["project"] == project) & (data["settings.preset_name"] == preset)]
            ticks.append(f"{project} {preset}")
            percentages.append(
                [
                    len(target_data[target_data["mutant_killed"]]) / len(target_data),
                    len(target_data[target_data["cosmic_ray_full.killed_by_own"]]) / len(target_data),
                    len(target_data[target_data["cosmic_ray_full.killed_by_any"]]) / len(target_data),
                ]
            )

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
    ax.yaxis.set_major_formatter(format_perecent())
    ax.legend(
        [bar3, bar2, bar1],
        [
            "Mutants killed by any test.",
            "Mutants killed by tests from the same preset.",
            "Mutants killed by sucessful runs.",
        ],
    )
    plt.show()


# %% Plot percentage of equivalence claims


@block
@savefig
def plot_percentage_of_equivalence_claims():
    ticks = []
    percentages_killed = []
    percentages_unkilled = []

    for preset in PRESETS:
        if preset == "baseline_without_iterations":
            continue

        for project in data["project"].unique():
            target_data = data[(data["project"] == project) & (data["preset"] == preset)]
            claims_unkilled = target_data[
                (target_data["outcome"] == EQUIVALENT) & ~target_data["cosmic_ray_full.killed_by_any"]
            ]
            claims_killed = target_data[
                (target_data["outcome"] == EQUIVALENT) & target_data["cosmic_ray_full.killed_by_any"]
            ]
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
    ax.yaxis.set_major_formatter(format_perecent())
    ax.legend(
        [bar1, bar2],
        ["Mutants claimed equivalent (not killed by any test)", "Mutants claimed equivalent (killed by some test)"],
    )
    return fig


# %% Plot percentage of failed runs


@block
@savefig
def plot_percentage_of_failed_runs():
    ticks = []
    percentages_killed = []
    percentages_unkilled = []

    for preset in data["settings.preset_name"].unique():
        if preset == "baseline_without_iterations":
            continue

        for project in data["project"].unique():
            target_data = data[(data["project"] == project) & (data["settings.preset_name"] == preset)]
            failed_runs_unkilled = target_data[
                (target_data["outcome"] == FAIL) & ~target_data["cosmic_ray_full.killed_by_any"]
            ]
            failed_runs_killed = target_data[
                (target_data["outcome"] == FAIL) & target_data["cosmic_ray_full.killed_by_any"]
            ]
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
    ax.yaxis.set_major_formatter(format_perecent())
    ax.legend(
        [bar1, bar2],
        ["Failed runs (mutant not killed by any test)", "Failed runs (mutant killed by some test)"],
    )
    return fig


# %% Write stats
# ======================================================================================================================


@block
def write_stats():
    cols = [
        "usage.cached_tokens",
        "usage.uncached_prompt_tokens",
        "usage.completion_tokens",
        "usage.cost.cached_tokens",
        "usage.cost.uncached_prompt_tokens",
        "usage.cost.completion_tokens",
        "usage.cost",
        "mutant_killed",
        "claimed_equivalent",
        "aborted",
        "cosmic_ray_full.killed_by_own",
        "cosmic_ray_full.killed_by_any",
        "exit_code_diff",
        "output_diff",
        "mutant_covered.by_experiment",
        "mutant_covered.by_test",
        "mutant_covered",
        "num_experiments",
        "num_tests",
        "num_turns",
        "num_turns_before_equivalence_claim",
        "num_completions",
        "num_completions_before_equivalence_claim",
        "num_invalid_experiments",
        "num_invalid_tests",
        "num_incomplete_responses",
        "num_experiment_import_errors",
        "num_test_import_errors",
    ]
    agg_stats = data[cols + ["preset", "project"]].copy()
    agg_stats["coverage.line_coverage_for_module"] = data["coverage.covered_line_ids"].map(len) / (
        data["coverage.covered_line_ids"].map(len) + data["coverage.missing_line_ids"].map(len)
    )
    agg_stats["coverage.branch_coverage_for_module"] = data["coverage.covered_branch_ids"].map(len) / (
        data["coverage.covered_branch_ids"].map(len) + data["coverage.missing_branch_ids"].map(len)
    )
    for col in cols:
        agg_stats[col] = agg_stats[col].astype("float64")

    agg_stats.fillna(0, inplace=True)
    Path("/mnt/temp/asdf.txt").write_text(
        agg_stats[(agg_stats["preset"] == "baseline_with_iterations") & (agg_stats["project"] == "flutes")]
        .groupby(["preset", "project"])
        .agg(AGGS)
        .to_string()
    )
    agg_stats.groupby("preset")[cols].agg(AGGS).transpose().to_csv(OUTPUT_PATH / 'data.groupby("preset").agg().csv')
    agg_stats.groupby(["preset", "project"]).sum().groupby("preset")[cols].agg(AGGS).transpose().to_csv(
        OUTPUT_PATH / 'data.groupby("preset", "project").sum().groupby("preset").agg().csv'
    )
    agg_stats.groupby(["preset", "project"])[cols].agg(AGGS).transpose().to_csv(
        OUTPUT_PATH / 'data.groupby("preset", "project").agg().csv'
    )


# %% Paper RQ 1: How does our approach perform compared to Pynguin?
# ======================================================================================================================


class ____RQ_1:  # mark this in the outline
    pass


# %% Usage (token and costs)
# ----------------------------------------------------------------------------------------------------------------------


class ____Usage:  # mark this in the outline
    pass


usage_cols = [
    "usage.cached_tokens",
    "usage.uncached_prompt_tokens",
    "usage.completion_tokens",
    "usage.cost.cached_tokens",
    "usage.cost.uncached_prompt_tokens",
    "usage.cost.completion_tokens",
    "usage.cost",
]


# %% Mean usage for a single mutant


data.groupby(["preset", "project"])[usage_cols].agg(AGGS).transpose().to_csv(
    OUTPUT_PATH / "token_usage_per_project.csv"
)

# %% Mean usage mean for a single project (1000 or fewer mutants)


data.groupby(["preset", "project"])[usage_cols].sum().groupby("preset").agg(AGGS).transpose().to_csv(
    OUTPUT_PATH / "token_usage_per_project.csv"
)

# %% Total usage


data[usage_cols].sum()


# %% Plot mean cost per preset and project


save_figs = True


@block
@savefig
def plot_mean_cost_2d():
    projects = sorted(list(data["project"].unique()))
    plot_data = data.groupby(["preset", "project"])

    x = []
    y = []
    s = []

    for y_, preset in enumerate(PRESETS[::-1]):
        for x_, project in enumerate(projects):
            group = plot_data.get_group((preset, project))
            x.append(x_)
            y.append(y_)
            s.append(group["usage.cost"].mean())

    s = np.array(s)
    s = s / s.max() * 1500

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 4))
    ax.scatter(x=x, y=y, s=s)

    ax.set_ylim((-0.5, 3.5))
    ax.set_yticks(range(4))
    ax.set_yticklabels(PRESETS[::-1])

    ax.set_xlim((-0.5, 9.5))
    ax.set_xticks(range(10))
    ax.set_xticklabels(projects, rotation=90)

    fig.legend(["Mean cost for each mutant"], loc="lower left")

    return fig


# %% Plot mean number of tokens for one mutant


@block
@savefig
def plot_num_tokens_per_mutant():
    return plot_bar_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        [
            ("Prompt tokens (cached)", lambda df: df["usage.cached_tokens"].mean()),
            ("Prompt tokens (uncached)", lambda df: df["usage.uncached_prompt_tokens"].mean()),
            ("Completion tokens", lambda df: df["usage.completion_tokens"].mean()),
        ],
    )


# %% Plot mean cost for one mutant


@block
@savefig
def plot_cost_per_mutant():
    return plot_bar_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        [
            ("Cost for prompt tokens (cached)", lambda df: df["usage.cost.cached_tokens"].mean()),
            ("Cost for prompt tokens (uncached)", lambda df: df["usage.cost.uncached_prompt_tokens"].mean()),
            ("Cost for completion tokens", lambda df: df["usage.cost.completion_tokens"].mean()),
        ],
    )


# %% Success rate
# ----------------------------------------------------------------------------------------------------------------------


class ____Success_Rate:  # mark this in the outline
    pass


# %% Plot percentage of outcomes


@block
@savefig
def plot_outcomes():
    return plot_bar_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        [
            ("Success", lambda df: (df["outcome"] == SUCCESS).mean()),
            ("Claimed Equivalent", lambda df: (df["outcome"] == EQUIVALENT).mean()),
            ("Failed", lambda df: (df["outcome"] == FAIL).mean()),
        ],
        customization=lambda fig, ax: ax.yaxis.set_major_formatter(format_perecent()),
    )


# %% Coverage
# ----------------------------------------------------------------------------------------------------------------------


class ____Coverage:  # mark this in the outline
    pass


# %% Plot mean line coverage


@block
@savefig
def plot_mean_line_coverage():
    def get_coverage_for_group(df):
        preset = df["preset"].unique()[0]
        projects = df["project"].unique()

        coverage_values = [coverage_map[(preset, project)] for project in projects]
        executed_lines = set()
        for c in coverage_values:
            executed_lines.update(c.executed_lines)
        return len(executed_lines) / sum([len(lines) for lines in all_seen_lines_map.values()])

    return plot_bar_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        [("Line coverage", get_coverage_for_group)],
        customization=lambda fig, ax: ax.yaxis.set_major_formatter(format_perecent()),
    )


# %% Plot mean branch coverage


@block
@savefig
def plot_mean_branch_coverage():
    def get_coverage_for_group(df):
        preset = df["preset"].unique()[0]
        projects = df["project"].unique()

        coverage_values = [coverage_map[(preset, project)] for project in projects]
        executed_branches = set()
        for c in coverage_values:
            executed_branches.update(c.executed_branches)
        return len(executed_branches) / sum([len(branches) for branches in all_seen_branches_map.values()])

    return plot_bar_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        [("Line coverage", get_coverage_for_group)],
        customization=lambda fig, ax: ax.yaxis.set_major_formatter(format_perecent()),
    )


# %% Plot mean line coverage with best Pynguin run

# TODO: combined pynguin coverage
# TODO: avg Pynguin coverage


@block
@savefig
def plot_mean_line_coverage_with_pynguin():
    values = []
    for preset in PRESETS:
        num_covered = 0
        for project in PROJECTS:
            num_covered += len(coverage_map[(preset, project)].executed_lines)
        values.append(num_covered)

    num_pynguin_covered = 0
    for project in PROJECTS:
        if is_pynguin_project_excluded(project):
            continue

        best_index = best_pynguin_runs[project].index_
        num_pynguin_covered += len(pynguin_coverage_map[(project, best_index)].executed_lines)
    values.append(num_pynguin_covered)

    values = np.array(values) / sum([len(branches) for branches in all_seen_lines_map.values()])

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.bar(x=np.arange(5), height=values)

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(PRESET_NAMES + ["Pynguin (best run)"], rotation=90)
    fig.legend(["Line Coverage"], loc="lower left")
    ax.yaxis.set_major_formatter(format_perecent())

    return fig


# %% Plot mean branch coverage with best Pynguin run


@block
@savefig
def plot_mean_branch_coverage_with_pynguin():
    values = []
    for preset in PRESETS:
        num_covered = 0
        for project in PROJECTS:
            num_covered += len(coverage_map[(preset, project)].executed_branches)
        values.append(num_covered)

    num_pynguin_covered = 0
    for project in PROJECTS:
        if is_pynguin_project_excluded(project):
            continue

        best_index = best_pynguin_runs[project].index_
        num_pynguin_covered += len(pynguin_coverage_map[(project, best_index)].executed_branches)
    values.append(num_pynguin_covered)

    values = np.array(values) / sum([len(branches) for branches in all_seen_branches_map.values()])

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.bar(x=np.arange(5), height=values)

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(PRESET_NAMES + ["Pynguin (best run)"], rotation=90)

    fig.legend(["Branch Coverage"], loc="lower left")
    ax.yaxis.set_major_formatter(format_perecent())
    return fig


# %% Mutation score
# ----------------------------------------------------------------------------------------------------------------------


class ____Mutation_Score:  # mark this in the outline
    pass


# %% Plot mean mutation scores


@block
@savefig
def plot_mean_mutation_scores():
    return plot_bar_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        [("Mutation Score", lambda df: df["cosmic_ray_full.killed_by_own"].sum() / len(df))],
        customization=lambda fig, ax: ax.yaxis.set_major_formatter(format_perecent()),
    )


# %% Plot mean mutation scores with best Pynguin run


@block
@savefig
def plot_mean_mutation_score_with_pynguin():
    values = []
    for preset in PRESETS:
        group = data[data["preset"] == preset]
        values.append(group["cosmic_ray_full.killed_by_own"].sum())

    def get_avg_kills_for_mutant(row):
        num_runs = get_num_pynguin_runs_per_project(row["project"])
        num_kills = row["pynguin.cosmic_ray.num_kills"]
        if num_runs == 0:
            if num_kills != 0:
                raise Exception("shouldn't happen")
            else:
                return 0
        return num_kills / num_runs

    # values.append(data[data["preset"] == PRESETS[0]]["pynguin.cosmic_ray.killed_by_best"].sum())
    values.append(data[data["preset"] == PRESETS[0]].apply(get_avg_kills_for_mutant, axis=1).sum())
    values.append(data[data["preset"] == PRESETS[0]]["pynguin.cosmic_ray.killed_by_any"].sum())

    values = np.array(values) / len(data[data["preset"] == PRESETS[0]])

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.bar(x=np.arange(6), height=values)

    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(PRESET_NAMES + ["Pynguin (average)", "Pynguin (combined)"], rotation=90)

    fig.legend(["Mutation Score"], loc="lower left")
    ax.yaxis.set_major_formatter(format_perecent())
    return fig


# %% Plot number of killed mutants with best Pynguin run


@block
@savefig
def plot_number_of_killed_mutants_with_pynguin():
    values = []
    for preset in PRESETS:
        group = data[data["preset"] == preset]
        values.append(group["cosmic_ray_full.killed_by_own"].sum())

    values.append(data[data["preset"] == PRESETS[0]]["pynguin.cosmic_ray.killed_by_best"].sum())
    values.append(data[data["preset"] == PRESETS[0]]["pynguin.cosmic_ray.killed_by_any"].sum())

    values = np.array(values)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.bar(x=np.arange(6), height=values)

    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(PRESET_NAMES + ["Pynguin (best run)", "Pynguin (combined)"], rotation=90)

    fig.legend(["Number of killed mutants"], loc="lower left")
    return fig


# %% Mann-Whitney U Test


@block
def mannwhitneyu_test():
    grouped_guut_data = data.groupby("preset")
    guut_groups = {preset: grouped_guut_data.get_group(preset)["cosmic_ray_full.killed_by_own"] for preset in PRESETS}

    grouped_pynguin_data = pynguin_data.groupby("index")  # pyright: ignore
    pynguin_groups = {index: grouped_pynguin_data.get_group(index)["killed"] for index in range(1, 31)}

    results = {}
    results["comparison"] = []
    for preset in PRESETS:
        results[f"{preset}_statistic"] = []
        results[f"{preset}_pvalue"] = []

    for group_name_2, values_2 in (guut_groups | pynguin_groups).items():
        if isinstance(group_name_2, int):
            group_name_2 = f"pynguin_{group_name_2}"
        results["comparison"].append(group_name_2)

        for group_name_1, values_1 in guut_groups.items():
            result = mannwhitneyu(values_1, values_2)
            results[f"{group_name_1}_statistic"].append(result.statistic)
            results[f"{group_name_1}_pvalue"].append(result.pvalue)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH / "mannwhitneyu.csv")


# %% Paper RQ 2: Can our approach reliably detect equivalent mutants
# ======================================================================================================================


class ____RQ_2:  # mark this in the outline
    pass


# %% Number of turns / completions
# ----------------------------------------------------------------------------------------------------------------------


class ____Number_Of_Turns:  # mark this in the outline
    pass


# %% Number of turns


@block
@savefig
def plot_num_turns():
    return plot_violin_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        "Number of turns (responses containing runnable code)",
        lambda df: df["num_turns"],
    )


# %% Number of turns (excluding turns after an equivalence claim is made)


# There are 356 (=len(data[data["mutant_killed"] & data["claimed_equivalent"]])) mutants that were claimed
# equivalent and then killed in the same conversation. This plot shows what the number of turns would look like
# if we stopped at every equivalence claim.

# The baselines start at 0 here, because baseline_with_iterations and baseline_without_iterations each had exactly
# one run, where the first response was invalid and the second response was an equivalence claim.


@block
@savefig
def plot_num_turns_before_equivalence_claim():
    return plot_violin_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        "Number of turns before equivalence claim",
        lambda df: df["num_turns_before_equivalence_claim"],
    )


# %% Number of completions


# The number of all messages generated by the LLM.
# This includes invalid responses and equivalence claims.


@block
@savefig
def plot_num_completions():
    return plot_violin_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        "Number of completions (all LLM-generated responses)",
        lambda df: df["num_completions"],
    )


# %% Number of completions (excluding completions after an equivalence claim is made)


# The number of all messages generated by the LLM.
# This includes invalid responses and equivalence claims.


@block
@savefig
def plot_num_completions_before_equivalence_claim():
    return plot_violin_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        "Number of completions before equivalence claim",
        lambda df: df["num_completions_before_equivalence_claim"],
    )


# %% Number of turns per outcome


@block
@savefig
def plot_num_turns_per_outcome():
    preset_data = data.groupby("preset")
    plots = []

    for preset, preset_name in zip(PRESETS, PRESET_NAMES):
        plot = plot_violin_per_group(
            preset_data.get_group(preset).groupby("outcome"),
            list(zip(OUTCOMES, OUTCOME_NAMES)),
            "Number of turns",
            lambda df: df["num_turns"],
            customization=lambda fig, ax: ax.set_title(preset_name),
        )
        plots.append(plot)

    return plots


# %% Number of turns per outcome (excluding turns after an equivalence claim is made)


@block
@savefig
def plot_num_turns_before_equivalence_claim_per_outcome():
    preset_data = data.groupby("preset")
    plots = []

    for preset, preset_name in zip(PRESETS, PRESET_NAMES):
        plot = plot_violin_per_group(
            preset_data.get_group(preset).groupby("outcome"),
            list(zip(OUTCOMES, OUTCOME_NAMES)),
            "Number of turns before equivalence claim",
            lambda df: df["num_turns_before_equivalence_claim"],
            customization=lambda fig, ax: ax.set_title(preset_name),
        )
        plots.append(plot)

    return plots


# %% Number of completions


@block
@savefig
def plot_num_completions_per_outcome():
    preset_data = data.groupby("preset")
    plots = []

    for preset, preset_name in zip(PRESETS, PRESET_NAMES):
        plot = plot_violin_per_group(
            preset_data.get_group(preset).groupby("outcome"),
            list(zip(OUTCOMES, OUTCOME_NAMES)),
            "Number of completions",
            lambda df: df["num_completions"],
            customization=lambda fig, ax: ax.set_title(preset_name),
        )
        plots.append(plot)

    return plots


# %% Number of completions


@block
@savefig
def plot_num_completions_before_equivalence_claim_per_outcome():
    preset_data = data.groupby("preset")
    plots = []

    for preset, preset_name in zip(PRESETS, PRESET_NAMES):
        plot = plot_violin_per_group(
            preset_data.get_group(preset).groupby("outcome"),
            list(zip(OUTCOMES, OUTCOME_NAMES)),
            "Number of completions before equivalence claim",
            lambda df: df["num_completions_before_equivalence_claim"],
            customization=lambda fig, ax: ax.set_title(preset_name),
        )
        plots.append(plot)

    return plots


# %% sandbox 2
# %% sandbox 3
# %% sandbox 4
