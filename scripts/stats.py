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
# - raw_pynguin_data
#     - Indexed by ("RunId", "TargetModule")
#     - Each row contains the results from running Pynguin on one module.
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
# - pynguin_num_tests_map
#     - Contains the number of Pynguin-generated tests for each EMSE project.
#     - Indexed by ("project", "index") where 1 <= index <= 30.
#
# - all_seen_lines_map / all_seen_branches_map
#     - Contains all (executed or missing) lines/branches from coverage.py for each EMSE project.
#     - Each entry contains a set of lines of one project.
#     - Indexed by "project"


# %% Imports and constants
# ======================================================================================================================

import ast
import itertools
import gzip
import json
import os
import sqlite3
import math
import re
from io import StringIO
from functools import partial, reduce
from multiprocessing import Pool
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Tuple,
    cast,
    Callable,
    Set,
)
from scipy.stats import mannwhitneyu
from collections import defaultdict

import plotly.graph_objects as go
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps as cm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from tqdm import tqdm

pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option(
    "display.float_format",
    lambda x: f"{int(x):d}" if math.floor(x) == x else f"{x:.3f}" if x > 1000 else f"{x:.6f}",
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

PRESET_NAMES_COMPACT = [
    "Baseline",
    "Iterative",
    "Scientific\n(0-shot)",
    "Scientific\n(1-shot)",
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
EQUIVALENT_SUCCESS = "equivalent+success"

OUTCOMES = [SUCCESS, EQUIVALENT, FAIL]
OUTCOME_NAMES = ["Success", "Claimed Equivalent", "Failed"]
FULL_OUTCOMES = [SUCCESS, EQUIVALENT_SUCCESS, EQUIVALENT, FAIL]
FULL_OUTCOME_NAMES = [
    "Success",
    "Claimed Equivalent and Killed",
    "Claimed Equivalent",
    "Failed",
]

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
            fig.savefig(OUTPUT_PATH / f"{function.__name__}.png", bbox_inches="tight")
            fig.savefig(OUTPUT_PATH / f"{function.__name__}.pdf", bbox_inches="tight")
        elif isinstance(retval, Tuple):
            fig, obj = retval
            fig.savefig(OUTPUT_PATH / f"{function.__name__}.png", bbox_inches="tight")
            fig.savefig(OUTPUT_PATH / f"{function.__name__}.pdf", bbox_inches="tight")
            (OUTPUT_PATH / f"{function.__name__}.json").write_text(json.dumps(obj))

        # elif isinstance(retval, Dict):
        #     fig = retval["fig"]
        #     if "bbox_extra" in retval:
        #         fig.savefig(OUTPUT_PATH / f"{function.__name__}.png", bbox_extra_artists=retval["bbox_extra"])
        #         fig.savefig(OUTPUT_PATH / f"{function.__name__}.pdf", bbox_extra_artists=retval["bbox_extra"])
        #     else:
        #         fig.savefig(OUTPUT_PATH / f"{function.__name__}.png")
        #         fig.savefig(OUTPUT_PATH / f"{function.__name__}.pdf")

        #     if "data" in retval:
        #         (OUTPUT_PATH / f"{function.__name__}.json").write_text(json.dumps(retval["data"]))

        # if isinstance(retval, list):
        #     for index, entry in enumerate(retval):
        #         fig = entry
        #         if isinstance(entry, Figure):
        #             fig = entry
        #             fig.savefig(OUTPUT_PATH / f"{function.__name__}_{index+1:02}.png")
        #             fig.savefig(OUTPUT_PATH / f"{function.__name__}.{index+1:02}.pdf")
        #         elif isinstance(entry, Tuple):
        #             fig, obj = entry
        #             fig.savefig(OUTPUT_PATH / f"{function.__name__}_{index+1:02}.png")
        #             fig.savefig(OUTPUT_PATH / f"{function.__name__}.{index+1:02}.pdf")
        #             (OUTPUT_PATH / f"{function.__name__}.json").write_text(json.dumps(obj))

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
        (
            "flake8",
            4,
        ),  # Tests cause pytest to raise an OSError. Most mutants are killed even though the tests pass.
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


# %% Flatten


def flatten(xss):
    return [x for xs in xss for x in xs]


# %% Parse info from the results directory name


class LongId(NamedTuple):
    preset: str
    package: str
    project: str
    id: str

    @staticmethod
    def parse_quoted(long_id: str) -> "LongId":
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


# %% Add escaped long id to data


FILENAME_REPLACEMENET_REGEX = r"[^0-9a-zA-Z]+"


def escape_path(name: str) -> str:
    return re.sub(FILENAME_REPLACEMENET_REGEX, "_", name)


@block
def add_clean_long_id_to_data():
    data["escaped_long_id"] = data["long_id"].map(escape_path)


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


# %% Compute full outcome


@block
def add_full_outcome_to_data():
    def get_full_outcome(row):
        if row["mutant_killed"] and row["claimed_equivalent"]:
            return EQUIVALENT_SUCCESS
        if row["mutant_killed"]:
            return SUCCESS
        elif row["claimed_equivalent"]:
            return EQUIVALENT
        else:
            return FAIL

    data["full_outcome"] = data.apply(get_full_outcome, axis=1).to_frame(name="full_outcome")


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


EMPTY_COVERAGE = Coverage(executed_branches=[], missing_branches=[], executed_lines=[], missing_lines=[])


def read_coverage_py_json(coverage_json: JsonObj) -> Coverage:
    missing_lines = []
    executed_lines = []
    missing_branches = []
    executed_branches = []
    for file_name, file in coverage_json["files"].items():
        if file_name.startswith("docstring_parser/tests/"):
            continue
        missing_lines += [LineId(file_name, line) for line in file["missing_lines"]]
        executed_lines += [LineId(file_name, line) for line in file["executed_lines"]]
        missing_branches += [BranchId(file_name, branch[0], branch[1]) for branch in file["missing_branches"]]
        executed_branches += [BranchId(file_name, branch[0], branch[1]) for branch in file["executed_branches"]]
    return Coverage(missing_lines, executed_lines, missing_branches, executed_branches)


def read_coverage():
    map = {}
    for coverage_path in RESULTS_DIR.glob("*/coverage/coverage.json"):
        results_dir = (coverage_path / ".." / "..").resolve()
        id = LongId.parse_quoted(results_dir.name)
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
        id = LongId.parse_quoted(results_dir.name)
        # cosmic_ray_results[(id.project, id.preset, "exitfirst")] = read_mutant_results(
        #     results_dir / "cosmic-ray" / "mutants.sqlite", id.project
        # )
        cosmic_ray_results[(id.project, id.preset, "full")] = read_mutant_results(
            results_dir / "cosmic-ray-full" / "mutants.sqlite", id.project
        )

    # data["cosmic_ray.killed_by_own"] = data.apply(
    #     lambda row: cosmic_ray_results[(row["project"], row["preset"], "exitfirst")][MutantId.from_row(row)].killed,
    #     axis=1,
    # )
    # presets = list(data["preset"].unique())
    # data["cosmic_ray.killed_by_any"] = data.apply(
    #     lambda row: any(
    #         cosmic_ray_results[(row["project"], preset, "exitfirst")][MutantId.from_row(row)].killed
    #         for preset in presets
    #     ),
    #     axis=1,
    # )

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


# %% Helper to get kill per test


def get_kills_per_test(df):
    kills_per_test = defaultdict(set)
    for index, row in df.iterrows():
        failing_tests = row["cosmic_ray_full.failing_tests"]
        mutant_id = row["mutant_id"]
        for test_id in failing_tests:
            kills_per_test[test_id].add(mutant_id)
    return kills_per_test


def get_pynguin_kills_per_test(df):
    kills_per_test = defaultdict(set)
    for index, row in df.iterrows():
        failing_tests = row["failing_tests"]
        mutant_id = row["mutant_id"]
        for test_id in failing_tests:
            kills_per_test[test_id].add(mutant_id)
    return kills_per_test


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
        failing_tests = get_failing_tests_for_pynguin_output(cosmic_ray_info.output)
        failing_tests = [f"{index}::{project}::{name}" for name in failing_tests]
        pynguin_data["failing_tests"].append(failing_tests)
pynguin_data = pd.DataFrame(pynguin_data)


# %% Sum up Pynguin results


type Project = str


class PynguinCosmicRayRunSummary(NamedTuple):
    index_: int
    kills: int


best_pynguin_runs: Dict[Project, PynguinCosmicRayRunSummary] = {}
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
        max_index, max_kills = best_pynguin_runs.get(project, (-1, -1))
        if num_kills > max_kills:
            best_pynguin_runs[project] = PynguinCosmicRayRunSummary(index, num_kills)


# %% Read raw Pynguin results

with gzip.open(REPO_PATH / "other_data" / "raw_pynguin_results.csv.gz") as g:
    raw_pynguin_data = pd.read_csv(StringIO(g.read().decode()))
raw_pynguin_data["TotalCost"] = (
    raw_pynguin_data["TotalTime"] / 1000000000 / 3600 * 0.0384
)  # 0.0384US-$/h fuer eine vergleichbare AWS-VM
raw_pynguin_data["Package"] = raw_pynguin_data["TargetModule"].map(lambda m: m.split(".")[0])
raw_pynguin_data["Project"] = raw_pynguin_data["Package"].map(lambda p: PACKAGE_TO_PROJECT.get(p, "unknown"))
raw_pynguin_data["Index"] = raw_pynguin_data["RandomSeed"] + 1
raw_pynguin_data["Excluded"] = raw_pynguin_data.apply(
    lambda row: is_pynguin_run_excluded(row["Project"], row["Index"]), axis=1
)


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
    lambda row: [f"{row['problem.target_path']}::{line}" for line in row["coverage.covered_lines"]],
    axis=1,
)
data["coverage.missing_line_ids"] = data.apply(
    lambda row: [f"{row['problem.target_path']}::{line}" for line in row["coverage.missing_lines"]],
    axis=1,
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
    lambda row: [
        f"{row['project']}::{row['problem.target_path']}::{branch}" for branch in row["coverage.covered_branches"]
    ],
    axis=1,
)
data["coverage.missing_branch_ids"] = data.apply(
    lambda row: [
        f"{row['project']}::{row['problem.target_path']}::{branch}" for branch in row["coverage.missing_branches"]
    ],
    axis=1,
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
        lambda row: all_lines.get((row["project"], row["problem.target_path"]), []),
        axis=1,
    )
    data["coverage.all_line_ids"] = data.apply(
        lambda row: all_line_ids.get((row["project"], row["problem.target_path"]), []),
        axis=1,
    )
    data["coverage.num_lines"] = data["coverage.all_lines"].map(len)

    data["coverage.all_branches"] = data.apply(
        lambda row: all_branches.get((row["project"], row["problem.target_path"]), []),
        axis=1,
    )
    data["coverage.all_branch_ids"] = data.apply(
        lambda row: all_branch_ids.get((row["project"], row["problem.target_path"]), []),
        axis=1,
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


# def estimate_loc(test):
#     if test is None:
#         return None
#     return len([line for line in test["code"].splitlines() if line.strip() and not line.strip().startswith("#")])


# data["test_loc"] = data["tests"].map(lambda tests: estimate_loc(find_killing_test(tests)))


# %% Load test LOC


loc_per_test = json.loads((REPO_PATH / "other_data" / "guut_loc_per_test.json").read_text())
loc_per_test = {Path(path).stem: loc for path, loc in loc_per_test.items()}

# %% Pynguin LOC


def prepare_pynguin_loc(loc):
    new_loc = {}
    for path, loc in loc.items():
        path = Path(path)
        parts = path.parts
        new_loc[f"{parts[1]}::{parts[0]}::{path.stem}"] = loc
    return new_loc


pynguin_loc_per_test = prepare_pynguin_loc(
    json.loads((REPO_PATH / "other_data" / "pynguin_loc_per_test_file.json").read_text())
)
pynguin_loc_per_test_minimized_individual = prepare_pynguin_loc(
    json.loads((REPO_PATH / "other_data" / "pynguin_loc_per_test_file_minimized_individual.json").read_text())
)
pynguin_loc_per_test_minimized_combined = prepare_pynguin_loc(
    json.loads((REPO_PATH / "other_data" / "pynguin_loc_per_test_file_minimized_combined.json").read_text())
)


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
            f"{len(sampled_mutants[sampled_mutants['mutant.num_equivalence_claims'] == num_claims]):4d} mutants with {num_claims} claims"
        )
    print()

    for project in data["project"].unique():
        print(f"{len(sampled_mutants[sampled_mutants['project'] == project]):4d} mutants from {project}")
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

    sample_data_map = {sample_row_to_mutant_id(row): sample_row_to_dict(row) for _, row in sample_data.iterrows()}

    for name in column_mapping.values():
        data[name] = data["mutant_id"].map(
            lambda mutant_id: convert_csv_value(sample_data_map[mutant_id][name])
            if mutant_id in sample_data_map
            else None
        )
        # data[name] = data[name].map(lambda val: None if np.isnan(val) else None)
    data["sampled"] = data["sample.equivalent"].map(lambda b: b is not None)

    return sample_data


sample_data = add_sample_results_to_data()


# %% Count number of final test cases to data


@block
def add_number_of_final_test_cases():
    def parse_number_of_test_cases(code: str | None):
        if code is None:
            return 0

        module = ast.parse(code, "test.py", "exec")
        top_level_func_defs = [
            node.name for node in module.body if isinstance(node, ast.FunctionDef) if node.name.startswith("test")
        ]
        return len(top_level_func_defs)

    data["num_final_test_cases"] = (
        data["tests"].map(find_killing_test).map(lambda x: x["code"] if x else None).map(parse_number_of_test_cases)
    )


# Number of test cases per test


def compute_num_cases_per_test():
    cases_per_test = {}

    for index, row in data.iterrows():
        cases_per_test[row["escaped_long_id"]] = row["num_final_test_cases"]

    return cases_per_test


num_cases_per_test = compute_num_cases_per_test()


# %% Count Pynguin test cases

PYNGUIN_TEST_DEF_REGEX = re.compile(r"^def test_case_\d+\(\)", re.MULTILINE)


def count_pynguin_test_cases():
    def parse_number_of_test_cases(code: str | None):
        if code is None:
            return 0
        return len(re.findall(PYNGUIN_TEST_DEF_REGEX, code))

    num_tests_per_exec = {}
    for project in PROJECTS:
        for index in range(1, 31):
            exec_dir = PYNGUIN_TESTS_DIR / project / f"{index:02d}"
            num_tests = 0

            if exec_dir.is_dir():
                for test_file in (exec_dir / "tests").iterdir():
                    if not test_file.is_file():
                        print(f"{test_file} is not a file")
                    else:
                        num_tests += parse_number_of_test_cases(test_file.read_text())

            num_tests_per_exec[(project, index)] = num_tests

    return num_tests_per_exec


pynguin_num_tests_map = count_pynguin_test_cases()

# %% Add number of direct kills to data


@block
def add_num_kills():
    kills = {}
    for index, row in data.iterrows():
        mutant_id = row["mutant_id"]
        if row["mutant_killed"]:
            kills[mutant_id] = kills.get(mutant_id, []) + [row["preset"]]

    data["mutant.kills"] = data.apply(
        lambda row: kills.get(row["mutant_id"], []),
        axis=1,
    )
    data["mutant.num_kills"] = data["mutant.kills"].map(len)


# %% Print memory usage


with open("/proc/self/status") as f:
    memusage = f.read().split("VmRSS:")[1].split("\n")[0][:-3]
    print(f"Memory used: {int(memusage.strip()) / (1024**2):.3f} GB")


# %% Print available columns (data_columns)


cols = sorted(list(data.columns))
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


def mix_colors(a, b, percentage=1):
    return [(a + (b * percentage)) / (1 + percentage) for a, b in zip(a, b)]


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
        target_data = cast(
            pd.DataFrame,
            data[(data["project"] == project) & (data["preset"] == preset)],
        )
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
            ax.vlines(
                x=lines,
                ymin=0.0,
                ymax=1.0,
                colors="lightgrey",
                ls="--",
                lw=1,
                label="modules",
            )

    fig.text(0.00, 1.02, project)
    ax.legend()
    plt.show()


# %% Number of experiments / tests per mutant


for preset in data["preset"].unique():
    if not preset.startswith("debugging"):
        continue

    target_data = data[data["preset"] == preset]

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
    ax.violinplot(
        [target_data["num_experiments"], target_data["num_tests"]],
        positions=[1, 2],
        showmeans=True,
    )
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
    print(f"  Successful: {len(runs[runs['outcome'] == 'success'])}")
    print(f"  Equivalent: {len(runs[runs['outcome'] == 'equivalent'])}")
    print(f"  Failed: {len(runs[runs['outcome'] == 'fail'])}")


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


print(f"Directly killed mutants: {len(data[data['mutant_killed']])}")
print(f"Mutants killed by tests from the same run: {len(data[data['cosmic_ray_full.killed_by_own']])}")
print(f"Mutants killed by tests from the any run: {len(data[data['cosmic_ray_full.killed_by_any']])}")


# %% Equivalent mutants


print(f"Equivalence-claimed mutants: {len(data[data['claimed_equivalent']])}")
print(
    f"Equivalence-claimed mutants not killed by test from same run: {len(data[data['claimed_equivalent'] & ~data['cosmic_ray_full.killed_by_own']])}"
)
print(
    f"Equivalence-claimed mutants not killed by test from any run: {len(data[data['claimed_equivalent'] & ~data['cosmic_ray_full.killed_by_any']])}"
)


# %% Number of mutants with 0/1/2/3 claims


print(f"mutants with {0} claims: {len(data[data['mutant.num_equivalence_claims'] == 0]) // 4:4d}")
print(f"mutants with {1} claims: {len(data[data['mutant.num_equivalence_claims'] == 1]) // 4:4d}")
print(f"mutants with {2} claims: {len(data[data['mutant.num_equivalence_claims'] == 2]) // 4:4d}")
print(f"mutants with {3} claims: {len(data[data['mutant.num_equivalence_claims'] == 3]) // 4:4d}")
print(f"mutants with {4} claims: {len(data[data['mutant.num_equivalence_claims'] == 4]) // 4:4d}")
print(f"total number of claims: {len(data[data['claimed_equivalent']])}")


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
        f"{project:<20} {len(claims_unkilled[claims_unkilled['mutant.num_equivalence_claims'] == 1]):3} {len(claims_unkilled[claims_unkilled['mutant.num_equivalence_claims'] == 2]) // 4:3} {len(claims_unkilled[claims_unkilled['mutant.num_equivalence_claims'] == 3]) // 4:3}"
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
        f"claims: {n}, equivalent: yes -> {len(target_data[target_data['sampled'] & (target_data['sample.equivalent'] == True) & (target_data['mutant.num_equivalence_claims'] == n)])}"
    )
    print(
        f"claims: {n}, equivalent: no -> {len(target_data[target_data['sampled'] & (target_data['sample.equivalent'] == False) & (target_data['mutant.num_equivalence_claims'] == n)])}"
    )
    print(
        f"claims: {n}, equivalent: yes, unsure -> {len(target_data[target_data['sampled'] & target_data['sample.unsure'] & (target_data['sample.equivalent'] == True) & (target_data['mutant.num_equivalence_claims'] == n)])}"
    )
    print(
        f"claims: {n}, equivalent: no, unsure -> {len(target_data[target_data['sampled'] & target_data['sample.unsure'] & (target_data['sample.equivalent'] == False) & (target_data['mutant.num_equivalence_claims'] == n)])}"
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
    groups: List[Tuple[Any, str]],
    cols: List[Tuple[str, Callable[[Any], Any]]],
    customization: Callable[[Any, Any], None] | None = None,
):
    values = [[query_group(grouped_data, group_key, fun, 0) for group_key, group_name in groups] for name, fun in cols]

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    for i in range(len(values)):
        ax.bar(x=np.arange(4), bottom=np.sum(values[:i], axis=0), height=values[i])

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([group_name for group_key, group_name in groups], rotation=90)

    if len(cols) == 1:
        ax.set_ylabel(cols[0][0])
    else:
        fig.legend([c[0] for c in cols], loc="lower left")

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
    **kwargs,
):
    values = [query_group(grouped_data, group_key, col_fun, [0]) for group_key, group_name in groups]

    # def add_noise(series):
    #         return series.to_numpy() + np.random.uniform(0, 1, len(series))
    # values = [add_noise(val) for val in values]

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    violin_parts = ax.violinplot(values, showmeans=True, widths=np.repeat(1, len(groups)), **kwargs)
    ax.margins(x=0.01)

    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = violin_parts[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)

    color = (cmap_colors[0][0] * 0.8, cmap_colors[0][1] * 0.8, cmap_colors[0][2] * 0.8)

    for vp in violin_parts["bodies"]:
        vp.set_facecolor(color)
        vp.set_edgecolor(color)
        vp.set_linewidth(1)
        vp.set_alpha(0.4)

    ax.set_xticks(np.arange(len(groups)) + 1)
    ax.set_xticklabels([group_name for group_key, group_name in groups], rotation=90)
    ax.set_ylabel(col_name)

    if customization:
        customization(fig, ax)

    # sns.stripplot(
    #     { x: add_noise(vals) for x, vals in zip(range(len(values)), values) },
    #     ax = ax,
    #     jitter = .5,
    #     size = 2,
    #     alpha = 0.3
    # )

    return fig


def plot_dist_per_group(
    grouped_data: Any,
    groups: List[Tuple[str, str]],
    col_name: str,
    col_fun: Callable[[Any], Any],
    customization: Callable[[Any, Any], None] | None = None,
):
    values = [list(query_group(grouped_data, group_key, col_fun, [0])) for group_key, group_name in groups]

    fig, ax = plt.subplots(layout="constrained", figsize=(6, 6))
    distribution_plot(values, ax=ax)

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([group_name for group_key, group_name in groups], rotation=90)
    ax.set_ylabel(col_name)

    if customization:
        customization(fig, ax)

    return fig


def mannwhitneyu_test(values, names, filename):
    results = {}
    results["comparison"] = []
    for name in names:
        results[f"{name}_statistic"] = []
        results[f"{name}_pvalue"] = []

    for group_name_2, values_2 in zip(names, values):
        results["comparison"].append(group_name_2)

        for group_name_1, values_1 in zip(names, values):
            result = mannwhitneyu(values_1, values_2)
            results[f"{group_name_1}_statistic"].append(result.statistic)
            results[f"{group_name_1}_pvalue"].append(result.pvalue)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH / filename)


def distribution_plot(
    data: pd.DataFrame | pd.Series | Dict[str, Any] | Any,
    *,
    x: Any | None = None,
    y: Any | None = None,
    hue: Any | None = None,
    order: (Any | None) = None,
    hue_order: (Any | None) = None,
    orient: Literal["v", "h", "x", "y"] | None = None,
    color: (Any | None) = None,
    palette: Any | None = None,
    width: float = 0.8,
    gap: float = 0.0,
    formatter: Callable[[Any], str] | None = None,
    ax: Any | None = None,
    **kwargs,
) -> None:
    """Draw a distribution plot, built of a boxplot and a stripplot.
    See the respective documentation for Seaborn's boxplot and stripplot for the various
    parameters and how the plots are created.
    Args:
        data: Dataset for plotting.  If ``x`` and ``y`` are absent, this is interpreted
              as wide-form.  Otherwise it is expected to be long-form.
        x: Name of a variable in ``data`` to be plotted on the x-axis.
        y: Name of a variable in ``data`` to be plotted on the y-axis.
        hue: Name of a variable in ``data`` that determines the hue of the colours.
        order: Order to plot the categorical levels in; otherwise the levels are
               inferred from the data objects.
        hue_order: Order to plot the categorical levels in; otherwise the levels are
                   inferred from the data objects.
        orient: Orientation of the plot (vertical or horizontal).  This is usually
                inferred based on the type of the input variables, but it can be used to
                resolve ambiguity when both ``x`` and ``y`` are numeric or when plotting
                wide-form data.
        color: Single colour for the elements in the plot.
        palette: Colours to use for the different levels of the ``hue`` variable.
                 Should be something that can be interpreted by ``color_palette()``, or
                 a dictionary mapping hue levels to matplotlib colours.
        width: Width allotted to each element on the orient axis.
        gap: Shrink on the orient axis by this factor to add a gap between dodged
             elements.
        formatter: Function for converting categorical data into strings.  Affects both
                   grouping and tick labels.
        ax: Axes object to draw the plot onto, otherwise use the current Axes.
        **kwargs: Other keyword arguments are passed through to the plot functions.
    """
    sns.boxplot(
        data,
        x=x,
        y=y,
        ax=ax,
        boxprops={"alpha": 0.4},
        color=color,
        fliersize=0.0,
        formatter=formatter,
        gap=gap,
        hue=hue,
        hue_order=hue_order,
        order=order,
        orient=orient,
        palette=palette,
        width=width,
        **kwargs,
    )
    sns.stripplot(
        data,
        x=x,
        y=y,
        ax=ax,
        color=color,
        dodge=False,
        formatter=formatter,
        hue=hue,
        hue_order=hue_order,
        order=order,
        orient=orient,
        palette=palette,
        size=2.5,
        **kwargs,
    )


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
    num_killed = []
    num_unkilled = []

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
            # percentages_killed.append(len(claims_killed) / len(target_data))
            # percentages_unkilled.append(len(claims_unkilled) / len(target_data))
            num_killed.append(len(claims_killed))
            num_unkilled.append(len(claims_unkilled))

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
    x = np.arange(len(ticks))
    bar1 = ax.bar(x, num_unkilled, color=cmap_colors[0])
    bar2 = ax.bar(
        x,
        num_killed,
        color=cmap_colors[1],
        bottom=num_unkilled,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, rotation=90)
    # ax.yaxis.set_major_formatter(format_perecent())
    ax.legend(
        [bar1, bar2],
        [
            "Mutants claimed equivalent (not killable with any generated test)",
            "Mutants claimed equivalent (killed by a generated test)",
        ],
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
        [
            "Failed runs (mutant not killed by any test)",
            "Failed runs (mutant killed by some test)",
        ],
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


# data.groupby(["preset", "project"])[usage_cols].agg(AGGS).transpose().to_csv(
#     OUTPUT_PATH / "token_usage_per_mutant.csv"
# )

# %% Mean usage mean for a single project (1000 or fewer mutants)


# data.groupby(["preset", "project"])[usage_cols].sum().groupby("preset").agg(AGGS).transpose().to_csv(
#     OUTPUT_PATH / "token_usage_per_project.csv"
# )

# %% Total usage


data[usage_cols].sum()


# %% Plot mean cost per preset and project


# @block
# @savefig
# def plot_mean_cost_2d():
#     projects = sorted(list(data["project"].unique()))
#     plot_data = data.groupby(["preset", "project"])

#     x = []
#     y = []
#     s = []

#     for y_, preset in enumerate(PRESETS[::-1]):
#         for x_, project in enumerate(projects):
#             group = plot_data.get_group((preset, project))
#             x.append(x_)
#             y.append(y_)
#             s.append(group["usage.cost"].mean())

#     s = np.array(s)
#     s = s / s.max() * 1500

#     fig, ax = plt.subplots(layout="constrained", figsize=(8, 4))
#     ax.scatter(x=x, y=y, s=s)

#     ax.set_ylim((-0.5, 3.5))
#     ax.set_yticks(range(4))
#     ax.set_yticklabels(PRESETS[::-1])

#     ax.set_xlim((-0.5, 9.5))
#     ax.set_xticks(range(10))
#     ax.set_xticklabels(projects, rotation=90)

#     fig.legend(["Mean cost for each mutant"], loc="lower left")

#     return fig


# %% Plot mean cost per preset and project as bars


# @block
# @savefig
def plot_mean_cost_per_project_():
    projects = sorted(list(data["project"].unique()))
    plot_data = data.groupby(["preset", "project"])

    mean_costs = [
        [plot_data.get_group((preset, project))["usage.cost"].mean() for project in PROJECTS] for preset in PRESETS
    ]

    fig, axs = plt.subplots(
        4,
        1,
        layout="constrained",
        figsize=(8, 6),
        sharex="all",
        sharey="all",
        squeeze=True,
    )

    for i in range(4):
        axs[i].bar(x=np.arange(10), height=mean_costs[i])
        axs[i].set_xlim((-0.5, 9.5))
        axs[i].set_xticks(range(10))
        axs[i].yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.1f}¢")
        axs[3].set_yticks([0.00, 0.002, 0.004, 0.006, 0.008, 0.010])
    axs[3].set_xticklabels(projects, rotation=90)

    fig.legend(["Mean cost for each mutant"], loc=(0.5, 0.03))

    return fig


# %% Plot cost per preset and project as bars


# @block
# @savefig
def plot_sum_cost_per_project_():
    projects = sorted(list(data["project"].unique()))
    plot_data = data.groupby(["preset", "project"])

    mean_costs = [
        [plot_data.get_group((preset, project))["usage.cost"].sum() for project in PROJECTS] for preset in PRESETS
    ]

    fig, axs = plt.subplots(
        4,
        1,
        layout="constrained",
        figsize=(8, 6),
        sharex="all",
        sharey="all",
        squeeze=True,
    )

    for i in range(4):
        axs[i].bar(x=np.arange(10), height=mean_costs[i])
        axs[i].set_xlim((-0.5, 9.5))
        axs[i].set_xticks(range(10))
        axs[i].yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}$")
    axs[3].set_yticks([0, 2, 4, 6, 8, 10])
    axs[3].set_xticklabels(projects, rotation=90)

    fig.legend(["Mean cost for each mutant"], loc=(0.5, 0.05))

    return fig


# %% Plot number of turns per preset and project as bars


@block
@savefig
def plot_sum_cost_per_project_with_avg_pynguin_cost():
    plot_data = data.groupby(["preset", "project"])
    costs = [[plot_data.get_group((preset, project))["num_turns"].mean() for project in PROJECTS] for preset in PRESETS]

    x = np.arange(len(PROJECTS)) * 6

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 6))

    ax.bar(x=x, height=costs[0], color=cmap_colors[0], width=1)
    ax.bar(x=x + 1, height=costs[1], color=cmap_colors[1], width=1)
    ax.bar(x=x + 2, height=costs[2], color=cmap_colors[2], width=1)
    ax.bar(x=x + 3, height=costs[3], color=cmap_colors[3], width=1)
    ax.set_xlim((-2, 60))
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}$")

    ax.set_xticks(x + 1.5)
    ax.set_xticklabels(PROJECTS, rotation=90)
    labels = PRESET_NAMES
    fig.legend(labels, loc=(0.1, 0.8))
    ax.set_ylabel("Cost for each project")

    return fig  # , list(zip(labels, [list(zip(PROJECTS, c)) for c in costs]))


# %% Plot total cost per preset and project as bars


@block
@savefig
def plot_sum_cost_per_project_with_total_pynguin_cost():
    plot_data = data.groupby(["preset", "project"])
    costs = [[plot_data.get_group((preset, project))["usage.cost"].sum() for project in PROJECTS] for preset in PRESETS]

    # Add Pynguin costs

    grouped_pynguin_data = raw_pynguin_data.groupby(["Project", "Index"])
    pynguin_costs = []
    for project in PROJECTS:
        cost_for_project = 0
        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue

            group = grouped_pynguin_data.get_group((project, index))
            cost_for_project += group["TotalCost"].sum()

        pynguin_costs.append(cost_for_project)  # / get_num_pynguin_runs_per_project(project))
    costs.append(pynguin_costs)

    x = np.arange(len(PROJECTS)) * 6

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 6))

    ax.bar(x=x, height=costs[0], color=cmap_colors[0], width=1)
    ax.bar(x=x + 1, height=costs[1], color=cmap_colors[1], width=1)
    ax.bar(x=x + 2, height=costs[2], color=cmap_colors[2], width=1)
    ax.bar(x=x + 3, height=costs[3], color=cmap_colors[3], width=1)
    ax.bar(x=x + 4, height=pynguin_costs, color=cmap_colors[4], width=1)
    ax.set_xlim((-2, 60))
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}$")

    ax.set_xticks(x + 2)
    ax.set_xticklabels(PROJECTS, rotation=90)
    labels = PRESET_NAMES + ["Pynguin (total)"]
    fig.legend(labels, loc=(0.1, 0.79))
    ax.set_ylabel("Cost for each project")

    plot_data = {}
    for label, preset_values in zip(labels, costs):
        plot_data[label] = {key: value for key, value in zip(PROJECTS, preset_values)}

    return fig, plot_data


# %% Plot total cost per preset and project as bars


@block
@savefig
def plot_mean_cost_per_project():
    plot_data = data.groupby(["preset", "project"])
    mean_costs = [
        [plot_data.get_group((preset, project))["usage.cost"].mean() for project in PROJECTS] for preset in PRESETS
    ]

    x = np.arange(len(PROJECTS)) * 5

    fig, ax = plt.subplots(layout="constrained", figsize=(8, 6))

    ax.bar(x=x, height=mean_costs[0], color=cmap_colors[0], width=1)
    ax.bar(x=x + 1, height=mean_costs[1], color=cmap_colors[1], width=1)
    ax.bar(x=x + 2, height=mean_costs[2], color=cmap_colors[2], width=1)
    ax.bar(x=x + 3, height=mean_costs[3], color=cmap_colors[3], width=1)
    ax.set_xlim((-2, 50))
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.1f}¢")

    ax.set_xticks(x + 1.5)
    ax.set_xticklabels(PROJECTS, rotation=90)
    fig.legend(PRESET_NAMES, loc=(0.1, 0.8))
    ax.set_ylabel("Mean cost per mutant")

    return fig, list(zip(PRESETS, [list(zip(PROJECTS, c)) for c in mean_costs]))


# %% Plot mean num tokens per mutant


@block
@savefig
def plot_num_tokens_per_mutant():
    grouped_data = data.groupby("preset")
    values = [
        [grouped_data.get_group(preset)[col].mean() for preset in PRESETS]
        for col in [
            "usage.cached_tokens",
            "usage.uncached_prompt_tokens",
            "usage.completion_tokens",
        ]
    ]

    colors = [
        cmap_colors[1],
        cmap_colors[0],
        cmap_colors[2],
    ]
    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    handles = []
    for i in range(len(values)):
        handles.append(
            ax.bar(
                x=np.arange(4),
                bottom=np.sum(values[:i], axis=0),
                height=values[i],
                color=colors[i],
            )
        )

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(PRESET_NAMES, rotation=90)
    labels = ["Prompt tokens (cached)", "Prompt tokens (uncached)", "Completion tokens"]
    fig.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(0.0, 0.1))
    ax.set_ylabel("Mean token usage per mutant")

    plot_data = {}
    for label, preset_values in zip(labels, values):
        plot_data[label] = {key: value for key, value in zip(PRESET_NAMES, preset_values)}

    return fig, plot_data


# %% Plot mean cost for one mutant


@block
@savefig
def plot_cost_per_mutant():
    grouped_data = data.groupby("preset")
    values = [
        [grouped_data.get_group(preset)[col].mean() for preset in PRESETS]
        for col in [
            "usage.cost.cached_tokens",
            "usage.cost.uncached_prompt_tokens",
            "usage.cost.completion_tokens",
        ]
    ]

    colors = [
        cmap_colors[1],
        cmap_colors[0],
        cmap_colors[2],
    ]

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    handles = []
    for i in range(len(values)):
        handles.append(
            ax.bar(
                x=np.arange(4),
                bottom=np.sum(values[:i], axis=0),
                height=values[i],
                color=colors[i],
            )
        )

    ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.1f}¢")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(PRESET_NAMES, rotation=90)
    ax.set_ylabel("Mean cost per mutant")
    labels = ["Prompt tokens (cached)", "Prompt tokens (uncached)", "Completion tokens"]
    fig.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(0.0, 0.1))

    plot_data = {}
    for label, preset_values in zip(labels, values):
        plot_data[label] = {key: value for key, value in zip(PRESET_NAMES, preset_values)}

    return fig, plot_data


# %% Mann Whitney U test for tokens per mutant


@block
def mannwhitneyu_tokens_per_mutant():
    grouped_data = data.groupby("preset")
    mannwhitneyu_test(
        [
            grouped_data.get_group(preset)["usage.cached_tokens"]
            + grouped_data.get_group(preset)["usage.uncached_prompt_tokens"]
            + grouped_data.get_group(preset)["usage.completion_tokens"]
            for preset in PRESETS
        ],
        PRESETS,
        "mannwhitneyu_plot_tokens_per_mutant.csv",
    )


# %% Mann Whitney U test for cost per mutant


@block
def mannwhitneyu_cost_per_mutant():
    grouped_data = data.groupby("preset")
    mannwhitneyu_test(
        [grouped_data.get_group(preset)["usage.cost"] for preset in PRESETS],
        PRESETS,
        "mannwhitneyu_plot_cost_per_mutant.csv",
    )


# %% Success rate
# ----------------------------------------------------------------------------------------------------------------------


class ____Success_Rate:  # mark this in the outline
    pass


# %% Plot percentage of outcomes


# @block
# @savefig
# def plot_outcomes():
#     grouped_data = data.groupby("preset")
#     values = [
#         [(grouped_data.get_group(preset)["outcome"] == outcome).mean() for preset in PRESETS]
#         for outcome in [SUCCESS, EQUIVALENT, FAIL]
#     ]

#     fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
#     for i in range(len(values)):
#         ax.bar(x=np.arange(4), bottom=np.sum(values[:i], axis=0), height=values[i])

#     ax.yaxis.set_major_formatter(format_perecent())
#     ax.set_xticks(np.arange(4))
#     ax.set_xticklabels(PRESET_NAMES, rotation=90)
#     labels = OUTCOME_NAMES
#     fig.legend(
#         labels,
#         loc="upper left",
#     )
#     ax.set_ylabel("Percentage of outcomes")
#     return fig, list(zip(labels, [list(zip(PRESET_NAMES, val)) for val in values]))


# %% Plot percentage of full outcomes


@block
@savefig
def plot_full_outcomes():
    grouped_data = data.groupby("preset")
    values = [
        [(grouped_data.get_group(preset)["full_outcome"] == outcome).mean() for preset in PRESETS]
        for outcome in FULL_OUTCOMES
    ]
    colors = [
        cmap_colors[4],
        cmap_colors[3],
        cmap_colors[2],
        cmap_colors[6],
    ]

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    handles = []
    for i in range(len(values)):
        handles.append(
            ax.bar(
                x=np.arange(4),
                bottom=np.sum(values[:i], axis=0),
                height=values[i],
                color=colors[i],
            )
        )

    ax.yaxis.set_major_formatter(format_perecent())
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(PRESET_NAMES, rotation=90)
    labels = FULL_OUTCOME_NAMES

    fig.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(-0.05, 0.1))

    ax.set_ylabel("Percentage of outcomes")

    plot_data = {}
    for label, preset_values in zip(labels, values):
        plot_data[label] = {key: value for key, value in zip(PRESET_NAMES, preset_values)}

    return fig, plot_data


# %% Plot percentage of outcomes per project


@block
@savefig
def plot_outcomes_by_project():
    grouped_data = data.groupby("project")
    values = [
        [(grouped_data.get_group(project)["outcome"] == outcome).mean() for project in PROJECTS] for outcome in OUTCOMES
    ]

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    for i in range(len(values)):
        ax.bar(
            x=np.arange(len(PROJECTS)),
            bottom=np.sum(values[:i], axis=0),
            height=values[i],
        )

    ax.yaxis.set_major_formatter(format_perecent())
    ax.set_xticks(np.arange(len(PROJECTS)))
    ax.set_xticklabels(PROJECTS, rotation=90)
    labels = OUTCOME_NAMES
    fig.legend(
        labels,
        loc="upper left",
    )
    ax.set_ylabel("Percentage of outcomes")
    return fig, list(zip(labels, [list(zip(PROJECTS, val)) for val in values]))


# %% Mann Whitney U test for mutant kills


@block
def mannwhitneyu_outcome_killed():
    grouped_data = data.groupby("preset")
    mannwhitneyu_test(
        [grouped_data.get_group(preset)["mutant_killed"] for preset in PRESETS],
        PRESETS,
        "mannwhitneyu_plot_outcome_killed.csv",
    )


# %% Mann Whitney U test for equivalence claims


@block
def mannwhitneyu_outcome_claimed_equivalent():
    grouped_data = data.groupby("preset")
    mannwhitneyu_test(
        [grouped_data.get_group(preset)["claimed_equivalent"] for preset in PRESETS],
        PRESETS,
        "mannwhitneyu_plot_outcome_claimed_equivalent.csv",
    )


# %% Coverage
# ----------------------------------------------------------------------------------------------------------------------


class ____Coverage:  # mark this in the outline
    pass


# %% Plot mean line coverage


# @block
@savefig
def plot_line_coverage():
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


# @block
@savefig
def plot_branch_coverage():
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


# %% Plot mean line coverage with Pynguin


@block
@savefig
def plot_line_coverage_with_pynguin_distplot():
    values = []
    for preset in PRESETS:
        num_covered = []
        for project in PROJECTS:
            num_covered.append(len(coverage_map[(preset, project)].executed_lines) / len(all_seen_lines_map[project]))
            # print(
            #     f"{len(coverage_map[(preset, project)].executed_lines) / len(all_seen_lines_map[project])} {preset} {project}"
            # )
        values.append(num_covered)

    # Pynguin combined runs
    num_combined_pynguin_covered = []
    for project in PROJECTS:
        if is_pynguin_project_excluded(project):
            continue

        covered_lines_per_project = set()

        for index in range(1, 31):
            coverage = pynguin_coverage_map.get((project, index), EMPTY_COVERAGE)
            covered_lines_per_project.update(coverage.executed_lines)

        num_combined_pynguin_covered.append(len(covered_lines_per_project) / len(all_seen_lines_map[project]))
    values.append(num_combined_pynguin_covered)

    # All Pynguin runs
    num_avg_pynguin_covered = []
    for project in PROJECTS:
        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue

            coverage = pynguin_coverage_map.get((project, index), EMPTY_COVERAGE)
            num_avg_pynguin_covered.append(len(coverage.executed_lines) / len(all_seen_lines_map[project]))
            # print(f"{len(coverage.executed_lines) / len(all_seen_lines_map[project])} {project} {index}")
    values.append(num_avg_pynguin_covered)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    labels = PRESET_NAMES + ["Pynguin (combined)", "Pynguin (individual)"]
    distribution_plot(values, ax=ax)

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(
        labels,
        rotation=90,
    )

    ax.yaxis.set_major_formatter(format_perecent())
    ax.set_ylabel("Line Coverage")

    mannwhitneyu_test(values, labels, "mannwhitneyu_plot_line_coverage_with_pynguin_distplot.csv")
    return fig, {key: value for key, value in zip(labels, [sum(val) / len(val) for val in values])}


# %% Plot mean branch coverage with Pynguin


@block
@savefig
def plot_branch_coverage_with_pynguin_distplot():
    values = []
    for preset in PRESETS:
        num_covered = []
        for project in PROJECTS:
            num_covered.append(
                len(coverage_map[(preset, project)].executed_branches) / len(all_seen_branches_map[project])
            )
            # print(
            #     f"{len(coverage_map[(preset, project)].executed_branches) / len(all_seen_branches_map[project])} {preset} {project}"
            # )
        values.append(num_covered)

    # Pynguin combined runs
    num_combined_pynguin_covered = []
    for project in PROJECTS:
        if is_pynguin_project_excluded(project):
            continue

        covered_branches_per_project = set()

        for index in range(1, 31):
            coverage = pynguin_coverage_map.get((project, index), EMPTY_COVERAGE)
            covered_branches_per_project.update(coverage.executed_branches)

        num_combined_pynguin_covered.append(len(covered_branches_per_project) / len(all_seen_branches_map[project]))
    values.append(num_combined_pynguin_covered)

    # All Pynguin runs
    num_avg_pynguin_covered = []
    for project in PROJECTS:
        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue

            coverage = pynguin_coverage_map.get((project, index), EMPTY_COVERAGE)
            num_avg_pynguin_covered.append(len(coverage.executed_branches) / len(all_seen_branches_map[project]))
            # print(f"{len(coverage.executed_branches) / len(all_seen_branches_map[project])} {project} {index}")
    values.append(num_avg_pynguin_covered)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    labels = PRESET_NAMES + ["Pynguin (combined)", "Pynguin (individual)"]
    distribution_plot(values, ax=ax)

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(
        labels,
        rotation=90,
    )

    ax.yaxis.set_major_formatter(format_perecent())
    ax.set_ylabel("Branch Coverage")

    mannwhitneyu_test(values, labels, "mannwhitneyu_plot_branch_coverage_with_pynguin_distplot.csv")
    return fig, {key: value for key, value in zip(labels, [sum(val) / len(val) for val in values])}


# %% Mutation score
# ----------------------------------------------------------------------------------------------------------------------


class ____Mutation_Score:  # mark this in the outline
    pass


# %% Plot mutation score coverage with Pynguin


@block
@savefig
def plot_mutation_score_with_pynguin_distplot():
    grouped_data = data.groupby(["preset", "project"])
    grouped_pynguin_data = pynguin_data.groupby(["project", "index"])

    values = []
    for preset in PRESETS:
        killed_mutants = []
        for project in PROJECTS:
            group = grouped_data.get_group((preset, project))
            killed_mutants.append(group["cosmic_ray_full.killed_by_own"].sum() / len(group))
            # print(f"{group['cosmic_ray_full.killed_by_own'].sum() / len(group)} {preset} {project}")
        values.append(killed_mutants)

    # Pynguin combined runs
    num_combined_pynguin = []
    for project in PROJECTS:
        if is_pynguin_project_excluded(project):
            continue

        group = grouped_data.get_group((PRESETS[0], project))
        num_combined_pynguin.append(group["pynguin.cosmic_ray.killed_by_any"].sum() / len(group))
    values.append(num_combined_pynguin)

    # All Pynguin runs
    num_all_pynguin = []
    for project in PROJECTS:
        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue

            group = grouped_pynguin_data.get_group((project, index))
            if group["killed"].sum() == 0:
                print(project, index)
            num_all_pynguin.append(group["killed"].sum() / len(group))
            # print(f"{group['killed'].sum() / len(group)} {project} {index}")
    values.append(num_all_pynguin)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    labels = PRESET_NAMES + ["Pynguin (combined)", "Pynguin (individual)"]
    distribution_plot(values, ax=ax)

    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(
        labels,
        rotation=90,
    )

    ax.yaxis.set_major_formatter(format_perecent())
    ax.set_ylabel("Mutation Score")

    mannwhitneyu_test(values, labels, "mannwhitneyu_plot_mutation_score_with_pynguin_distplot.csv")
    return fig, {key: value for key, value in zip(labels, [sum(val) / len(val) for val in values])}


# %% Mann Whitney U test with values from mean mutation scores plot


@block
def mutation_scores_mannwhitneyu_test():
    values = []
    for preset in PRESETS:
        group = data[data["preset"] == preset]
        values.append(list(group["cosmic_ray_full.killed_by_own"]))

    def get_avg_kills_for_mutant(row):
        num_runs = get_num_pynguin_runs_per_project(row["project"])
        num_kills = row["pynguin.cosmic_ray.num_kills"]
        if num_runs == 0:
            if num_kills != 0:
                raise Exception("shouldn't happen")
            else:
                return 0
        return num_kills / num_runs

    values.append(list(data[data["preset"] == PRESETS[0]]["pynguin.cosmic_ray.killed_by_any"]))
    values.append(list(data[data["preset"] == PRESETS[0]].apply(get_avg_kills_for_mutant, axis=1)))

    names = PRESETS + ["pynguin_combined", "pynguin_avg"]
    mannwhitneyu_test(values, names, "mannwhitneyu_mutation_score_individual_with_pynugin.csv")


# %% Mann-Whitney U Test


@block
def killed_mutants_mannwhitneyu_test():
    grouped_guut_data = data.groupby("preset")
    guut_groups = {preset: grouped_guut_data.get_group(preset)["cosmic_ray_full.killed_by_own"] for preset in PRESETS}

    grouped_pynguin_data = pynguin_data.groupby("index")  # pyright: ignore
    pynguin_groups = [grouped_pynguin_data.get_group(index)["killed"] for index in range(1, 31)]

    names = PRESETS + [f"pynguin_{i}" for i in range(30)]
    values = [guut_groups[p] for p in PRESETS] + pynguin_groups

    mannwhitneyu_test(values, names, "mannwhitneyu_killed_mutants.csv")


# %% Number of tests
# ----------------------------------------------------------------------------------------------------------------------


class ____Number_Of_Tests:  # mark this in the outline
    pass


# %% Plot mean number of tests per mutant with Pynguin


@block
@savefig
def plot_number_of_test_cases_distplot():
    grouped_data = data.groupby(["preset", "project"])

    values = []
    for preset in PRESETS:
        num_tests_per_project = []

        for project in PROJECTS:
            group = grouped_data.get_group((preset, project))
            num_tests_per_project.append(group["num_final_test_cases"].sum())

        values.append([int(n) for n in num_tests_per_project])

    # Pynguin runs
    num_all_pynguin_tests = []
    for project in PROJECTS:
        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue

            num_tests = pynguin_num_tests_map.get((project, index), 0)
            num_all_pynguin_tests.append(num_tests)
    values.append(num_all_pynguin_tests)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    labels = PRESET_NAMES + ["Pynguin (individual)"]  # , "Pynguin (combined)"]
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Number of tests")
    distribution_plot(values, ax=ax)

    mannwhitneyu_test(values, labels, "mannwhitneyu_plot_number_of_test_cases_distplot.csv")

    # Pynguin runs
    num_combined_pynguin_tests = []
    for project in PROJECTS:
        acc = 0
        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue

            acc += pynguin_num_tests_map.get((project, index), 0)

        num_combined_pynguin_tests.append(acc)
    values.append(num_combined_pynguin_tests)
    labels.append("Pynguin (combined)")
    return fig, {key: value for key, value in zip(labels, [sum(val) / len(val) for val in values])}


# %% Plot mean number of loc per project


@block
@savefig
def plot_loc_distplot():
    grouped_data = data.groupby(["preset", "project"])

    values = []
    for preset in PRESETS:
        num_loc_per_project = []

        for project in PROJECTS:
            group = grouped_data.get_group((preset, project))
            num_loc_per_project.append(sum([loc_per_test.get(id) or 0 for id in group["escaped_long_id"]]))

        values.append([int(n) for n in num_loc_per_project])

    pynguin_individual_values = []
    for project in PROJECTS:
        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue

            project_loc = 0
            for key, loc in pynguin_loc_per_test.items():
                if key.startswith(f"{index:02}::{project}"):
                    project_loc += loc

            # print(f"{project_loc} {project} {index}")
            pynguin_individual_values.append(project_loc)
    values.append(pynguin_individual_values)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    labels = PRESET_NAMES + ["Pynguin (individual)"]  # , "Pynguin (combined)"]
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Lines of code")
    distribution_plot(values, ax=ax)

    mannwhitneyu_test(values, labels, "mannwhitneyu_loc_distplot.csv")

    pynguin_combined_values = []
    for project in PROJECTS:
        acc = 0
        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue

            project_loc = 0
            for key, loc in pynguin_loc_per_test.items():
                if key.startswith(f"{index:02}::{project}"):
                    acc += loc
        pynguin_combined_values.append(acc)
    values.append(pynguin_combined_values)
    labels.append("Pynguin (combined)")
    return fig, {key: value for key, value in zip(labels, [sum(val) / len(val) for val in values])}


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
    plot_data = data.groupby("preset")["num_turns"].agg(AGGS)
    fig = plot_violin_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        "Number of iterations",
        lambda df: df["num_turns"],
    )

    plot_data = plot_data.transpose().to_dict()
    preset_to_names = {preset: name for preset, name in zip(PRESETS, PRESET_NAMES)}
    plot_data = {preset_to_names[preset]: values for preset, values in plot_data.items()}
    return fig, plot_data


# %% Number of turns for a successful test


@block
@savefig
def plot_num_turns_success():
    fig = plot_violin_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        "Number of iterations",
        lambda df: df[df["mutant_killed"]]["num_turns"],
    )

    plot_data = data[data["mutant_killed"]].groupby("preset")["num_turns"].agg(AGGS).transpose().to_dict()
    preset_to_names = {preset: name for preset, name in zip(PRESETS, PRESET_NAMES)}
    plot_data = {preset_to_names[preset]: values for preset, values in plot_data.items()}
    return fig, plot_data


# %% Number of turns for a successful test


@block
@savefig
def plot_num_turns_fail():
    plot_data = data.groupby("preset")["num_turns"].agg(AGGS)
    return plot_violin_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        "Number of iterations",
        lambda df: df[df["outcome"] == FAIL]["num_turns"],
    ), plot_data.transpose().to_dict()


# %% Number of turns for a successful test


@block
@savefig
def plot_num_turns_equivalent():
    plot_data = data.groupby("preset")["num_turns"].agg(AGGS)
    return plot_violin_per_group(
        data.groupby("preset"),
        list(zip(PRESETS[1:], PRESET_NAMES[1:])),
        "Number of iterations",
        lambda df: df[df["claimed_equivalent"]]["num_turns"],
    ), plot_data.transpose().to_dict()


# %% Mann Whitney U test for number of turns


@block
def mannwhitneyu_plot_num_turns():
    plot_data = data.groupby("preset")["num_turns"]
    mannwhitneyu_test(
        [plot_data.get_group(preset) for preset in PRESETS],
        PRESET_NAMES,
        "mannwhitneyu_plot_num_turns.csv",
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
        list(zip(PRESETS[1:], PRESET_NAMES[1:])),
        "Number of turns before equivalence claim",
        lambda df: df[df["claimed_equivalent"]]["num_turns_before_equivalence_claim"].map(lambda n: n + 1),
        points=10,
    )


# %% Mann Whitney U test for number of completio10


@block
def mannwhitneyu_plot_num_completions():
    plot_data = data.groupby("preset")["num_completions"]
    mannwhitneyu_test(
        [plot_data.get_group(preset) for preset in PRESETS],
        PRESET_NAMES,
        "mannwhitneyu_plot_num_completions.csv",
    )


# %% Number of completions


# The number of all messages generated by the LLM.
# This includes invalid responses and equivalence claims.


@block
@savefig
def plot_num_completions():
    plot_data = data.groupby("preset")["num_completions"].agg(AGGS)
    return plot_violin_per_group(
        data.groupby("preset"),
        list(zip(PRESETS, PRESET_NAMES)),
        "Number of completions",
        lambda df: df["num_completions"],
    ), plot_data.transpose().to_dict()


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


# @block
# @savefig
# def plot_num_turns_per_outcome():
#     preset_data = data.groupby("preset")
#     plots = []
#
#     for preset, preset_name in zip(PRESETS, PRESET_NAMES):
#         plot = plot_violin_per_group(
#             preset_data.get_group(preset).groupby("outcome"),
#             list(zip(OUTCOMES, OUTCOME_NAMES)),
#             "Number of turns",
#             lambda df: df["num_turns"],
#             customization=lambda fig, ax: ax.set_title(preset_name),
#         )
#         plots.append(plot)
#
#     return plots


# %% Number of turns per outcome (excluding turns after an equivalence claim is made)


# @block
# @savefig
# def plot_num_turns_before_equivalence_claim_per_outcome():
#     preset_data = data.groupby("preset")
#     plots = []
#
#     for preset, preset_name in zip(PRESETS, PRESET_NAMES):
#         plot = plot_violin_per_group(
#             preset_data.get_group(preset).groupby("outcome"),
#             list(zip(OUTCOMES, OUTCOME_NAMES)),
#             "Number of turns before equivalence claim",
#             lambda df: df["num_turns_before_equivalence_claim"],
#             customization=lambda fig, ax: ax.set_title(preset_name),
#         )
#         plots.append(plot)
#
#     return plots


# %% Number of completions


# @block
# @savefig
# def plot_num_completions_per_outcome():
#     preset_data = data.groupby("preset")
#     plots = []
#
#     for preset, preset_name in zip(PRESETS, PRESET_NAMES):
#         plot = plot_violin_per_group(
#             preset_data.get_group(preset).groupby("outcome"),
#             list(zip(OUTCOMES, OUTCOME_NAMES)),
#             "Number of completions",
#             lambda df: df["num_completions"],
#             customization=lambda fig, ax: ax.set_title(preset_name),
#         )
#         plots.append(plot)
#
#     return plots


# %% Number of completions


# @block
# @savefig
# def plot_num_completions_before_equivalence_claim_per_outcome():
#     preset_data = data.groupby("preset")
#     plots = []
#
#     for preset, preset_name in zip(PRESETS, PRESET_NAMES):
#         plot = plot_violin_per_group(
#             preset_data.get_group(preset).groupby("outcome"),
#             list(zip(OUTCOMES, OUTCOME_NAMES)),
#             "Number of completions before equivalence claim",
#             lambda df: df["num_completions_before_equivalence_claim"],
#             customization=lambda fig, ax: ax.set_title(preset_name),
#         )
#         plots.append(plot)
#
#     return plots


# %% Paper RQ 3: Can our approaches reliably detect equivalent mutants
# ======================================================================================================================


class ____RQ_3:  # mark this in the outline
    pass


# Print number of killed / unkilled equivalent mutants


@block
def print_num_equivalent_mutants():
    df = data[data["preset"] == PRESETS[0]]

    # %% Number of mutants
    print("all mutants", df["mutant_id"].count())

    # %% Number flagged mutants
    print(
        "claimed mutants",
        df[(df["mutant.num_equivalence_claims"] > 0)]["mutant_id"].count(),
    )

    # %% Number of unkilled flagged mutants (direct)
    print(
        "claimed and not killed in run",
        df[(df["mutant.num_equivalence_claims"] > 0) & (df["mutant.num_kills"] == 0)]["mutant_id"].count(),
    )

    # %% Number of unkilled flagged mutants (direct + cosmic_ray)
    print(
        "claimed and not killed in run or with cosmic_ray",
        df[
            (df["mutant.num_equivalence_claims"] > 0)
            & (df["mutant.num_kills"] == 0)
            & (~df["cosmic_ray_full.killed_by_any"])
        ]["mutant_id"].count(),
    )

    # %% Number of unkilled mutants (LLM-generated tests + Pynguin tests)
    print(
        "claimed and not killed in run or with cosmic-ray or with Pynguin cosmic-ray",
        df[
            (df["mutant.num_equivalence_claims"] > 0)
            & (df["mutant.num_kills"] == 0)
            & (~df["cosmic_ray_full.killed_by_any"])
            & (~df["pynguin.cosmic_ray.killed_by_any"])
        ]["mutant_id"].count(),
    )


# %% Write stats about sampled mutants.


@block
def write_sampled_mutants_tables():
    data[(data["preset"] == PRESETS[0]) & data["sampled"]][
        ["sample.equivalent", "sample.unsure", "mutant.num_equivalence_claims"]
    ].map(int).agg(["sum", "mean", "count"]).to_csv(OUTPUT_PATH / "sampled_mutants_results.csv")

    data[(data["preset"] == PRESETS[0]) & data["sampled"]][
        ["sample.equivalent", "sample.unsure", "mutant.num_equivalence_claims"]
    ].map(int).groupby("mutant.num_equivalence_claims").agg(["sum", "mean", "count"]).to_csv(
        OUTPUT_PATH / "sampled_mutants_results_per_num_claims.csv"
    )


# %% Plot number of killed and unkilled equivalent mutants


@block
@savefig
def plot_number_of_killed_and_unkilled_equivalences():
    num_killed = []
    num_unkilled = []

    plot_data = data.groupby("preset")
    for preset in PRESETS:
        if preset == "baseline_without_iterations":
            continue

        group = plot_data.get_group(preset)
        # don't use outcome for this
        claims_unkilled = group[(group["outcome"] == EQUIVALENT) & ~group["cosmic_ray_full.killed_by_any"]]
        claims_killed = group[(group["outcome"] == EQUIVALENT) & group["cosmic_ray_full.killed_by_any"]]
        num_killed.append(len(claims_killed))
        num_unkilled.append(len(claims_unkilled))

    fig, ax = plt.subplots(layout="constrained", figsize=(3, 6))
    x = np.arange(3)
    bar1 = ax.bar(x, num_unkilled, color=cmap_colors[0])
    bar2 = ax.bar(
        x,
        num_killed,
        color=cmap_colors[1],
        bottom=num_unkilled,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(PRESET_NAMES[1:], rotation=90)
    fig.legend(
        [
            "alive",
            "killed",
        ],
        loc="lower left",
    )
    return fig


# %% sandbox 2

data[(data["preset"] == PRESETS[0]) & data["sampled"]].groupby("project")["id"].count()
# %% sandbox 3

data.groupby(["preset", "project"])["usage.cost"].sum()


# %% sandbox 4


raw_pynguin_data[(~raw_pynguin_data["Excluded"]) & (raw_pynguin_data["Project"].isin(PROJECTS))].groupby(
    ["Project", "Index"]
)["TotalCost"].sum().groupby("Project").sum()

# %% sandbox 5

raw_pynguin_data[(~raw_pynguin_data["Excluded"]) & (raw_pynguin_data["Project"].isin(PROJECTS))].groupby(
    ["Project", "Index"]
)["TotalCost"].sum().groupby("Project").mean()


# %% Test Suite Minimization
# ======================================================================================================================


class ____Test_Suite_Minimization:  # mark this in the outline
    pass


# %% Minimize test suites


def minimize_test_suite_by_ms(df):
    kills_per_test = get_kills_per_test(df)
    tests_with_kills = list(kills_per_test.items())
    tests_with_kills = sorted(tests_with_kills, key=lambda x: len(x[1]), reverse=True)

    selected_tests = set()
    seen_killed_mutants = set()
    for test, kills in tests_with_kills:
        if kills.issubset(seen_killed_mutants):
            continue

        selected_tests.add(test)
        seen_killed_mutants.update(kills)

    return selected_tests


def minimize_pynguin_test_suite_by_ms(df):
    kills_per_test = get_pynguin_kills_per_test(df)
    tests_with_kills = list(kills_per_test.items())
    tests_with_kills = sorted(tests_with_kills, key=lambda x: len(x[1]), reverse=True)

    selected_tests = set()
    seen_killed_mutants = set()
    for test, kills in tests_with_kills:
        if kills.issubset(seen_killed_mutants):
            continue

        selected_tests.add(test)
        seen_killed_mutants.update(kills)

    return selected_tests


def minimize_test_suite_by_line_coverage(df):
    tests_with_coverage = [(row["escaped_long_id"], set(row["coverage.covered_line_ids"])) for _, row in df.iterrows()]
    tests_with_coverage = sorted(tests_with_coverage, key=lambda x: len(x[1]), reverse=True)

    selected_tests = set()
    seen_covered_lines = set()
    for test, covered_lines in tests_with_coverage:
        if covered_lines.issubset(seen_covered_lines):
            continue

        selected_tests.add(test)
        seen_covered_lines.update(covered_lines)

    return selected_tests


def minimize_test_suite_by_branch_coverage(df):
    tests_with_coverage = [
        (row["escaped_long_id"], set(row["coverage.covered_branch_ids"])) for _, row in df.iterrows()
    ]
    tests_with_coverage = sorted(tests_with_coverage, key=lambda x: len(x[1]), reverse=True)

    selected_tests = set()
    seen_covered_branches = set()
    for test, covered_branches in tests_with_coverage:
        if covered_branches.issubset(seen_covered_branches):
            continue

        selected_tests.add(test)
        seen_covered_branches.update(covered_branches)

    return selected_tests


def get_kills(group, suite):
    kills_per_test = get_kills_per_test(group)
    killed_mutants = set()

    for test in suite:
        killed_mutants.update(kills_per_test[test])

    return killed_mutants


class TestSuiteWithKills(NamedTuple):
    tests: Set[str]
    kills: Set[MutantId]


class MinimizedTestSuites(NamedTuple):
    tests_unminimized: TestSuiteWithKills
    tests_minimized_by_mutation_score: TestSuiteWithKills
    tests_minimized_by_branch_coverage: TestSuiteWithKills | None
    tests_minimized_by_line_coverage: TestSuiteWithKills | None


def minimize_test_suites():
    minimized_suites = {}

    for preset, project in itertools.product(PRESETS, PROJECTS):
        group = data[(data["preset"] == preset) & (data["project"] == project)]

        large_suite = set(group[group["mutant_killed"]]["escaped_long_id"])
        large_suite_kills = get_kills(group, large_suite)

        minimized_suite_by_ms = minimize_test_suite_by_ms(group)
        minimized_suite_by_ms_kills = get_kills(group, minimized_suite_by_ms)

        minimized_suite_by_line_coverage = minimize_test_suite_by_line_coverage(group)
        minimized_suite_by_line_coverage_kills = get_kills(group, minimized_suite_by_line_coverage)

        minimized_suite_by_branch_coverage = minimize_test_suite_by_branch_coverage(group)
        minimized_suite_by_branch_coverage_kills = get_kills(group, minimized_suite_by_branch_coverage)

        minimized_suites[(preset, project)] = MinimizedTestSuites(
            TestSuiteWithKills(large_suite, large_suite_kills),
            TestSuiteWithKills(minimized_suite_by_ms, minimized_suite_by_ms_kills),
            TestSuiteWithKills(minimized_suite_by_line_coverage, minimized_suite_by_line_coverage_kills),
            TestSuiteWithKills(
                minimized_suite_by_branch_coverage,
                minimized_suite_by_branch_coverage_kills,
            ),
        )

        print(f"\n{preset} / {project}")
        print(f"unminimized test files: {len(large_suite)}")
        print(f"unminimized kills: {len(large_suite_kills)}")
        # print(f"unminimized test cases: {sum([num_cases_per_test[test] for test in large_suite])}")
        print(f"minimized by MS test files: {len(minimized_suite_by_ms)}")
        print(f"minimized by MS kills: {len(minimized_suite_by_ms_kills)}")
        # print(f"minimized by MS test cases: {sum([num_cases_per_test[test] for test in minimized_suite])}")
        print(f"minimized by branch coverage test files: {len(minimized_suite_by_branch_coverage)}")
        print(f"minimized by branch coverage kills: {len(minimized_suite_by_branch_coverage_kills)}")
        print(f"minimized by line coverage test files: {len(minimized_suite_by_line_coverage)}")
        print(f"minimized by line coverage kills: {len(minimized_suite_by_line_coverage_kills)}")

    return minimized_suites


minimized_test_suites = minimize_test_suites()

# %% Minimize Pynguin Test Suites


def get_pynguin_kills(group, suite):
    kills_per_test = get_pynguin_kills_per_test(group)
    killed_mutants = set()

    for test in suite:
        killed_mutants.update(kills_per_test[test])

    return killed_mutants


def minimize_pynguin_test_suites():
    minimized_suites = {}

    for project, index in itertools.product(PROJECTS, list(range(1, 31))):
        if is_pynguin_run_excluded(project, index):
            continue

        group = pynguin_data[(pynguin_data["project"] == project) & (pynguin_data["index"] == index)]

        large_suite = set(flatten(group["failing_tests"]))
        large_suite_kills = get_pynguin_kills(group, large_suite)

        minimized_suite_by_ms = minimize_pynguin_test_suite_by_ms(group)
        minimized_suite_by_ms_kills = get_pynguin_kills(group, minimized_suite_by_ms)

        minimized_suites[(project, index)] = MinimizedTestSuites(
            TestSuiteWithKills(large_suite, large_suite_kills),
            TestSuiteWithKills(minimized_suite_by_ms, minimized_suite_by_ms_kills),
            None,
            None,
        )

        print(f"\n{project} / {index}")
        print(f"unminimized test cases: {len(large_suite)}")
        print(f"unminimized kills: {len(large_suite_kills)}")
        # print(f"unminimized test cases: {sum([num_cases_per_test[test] for test in large_suite])}")
        print(f"minimized by MS test cases: {len(minimized_suite_by_ms)}")
        print(f"minimized by MS kills: {len(minimized_suite_by_ms_kills)}")
        # print(f"minimized by MS test cases: {sum([num_cases_per_test[test] for test in minimized_suite])}")

    return minimized_suites


minimized_pynguin_test_suites = minimize_pynguin_test_suites()

# %% Minimized combined Pynguin test suites


def minimize_combined_pynguin_test_suites():
    minimized_suites = {}

    for project in PROJECTS:
        if is_pynguin_project_excluded(project):
            continue

        group = pynguin_data[pynguin_data["project"] == project]

        large_suite = set(flatten(group["failing_tests"]))
        large_suite_kills = get_pynguin_kills(group, large_suite)

        minimized_suite_by_ms = minimize_pynguin_test_suite_by_ms(group)
        minimized_suite_by_ms_kills = get_pynguin_kills(group, minimized_suite_by_ms)

        minimized_suites[project] = MinimizedTestSuites(
            TestSuiteWithKills(large_suite, large_suite_kills),
            TestSuiteWithKills(minimized_suite_by_ms, minimized_suite_by_ms_kills),
            None,
            None,
        )

        print(f"\n{project}")
        print(f"unminimized test cases: {len(large_suite)}")
        print(f"unminimized kills: {len(large_suite_kills)}")
        # print(f"unminimized test cases: {sum([num_cases_per_test[test] for test in large_suite])}")
        print(f"minimized by MS test cases: {len(minimized_suite_by_ms)}")
        print(f"minimized by MS kills: {len(minimized_suite_by_ms_kills)}")
        # print(f"minimized by MS test cases: {sum([num_cases_per_test[test] for test in minimized_suite])}")

    return minimized_suites


minimized_combined_pynguin_test_suites = minimize_combined_pynguin_test_suites()


# %% Plot number of tests cases in minimized test suites


@block
@savefig
def plot_number_of_minimized_test_cases_distplot():
    values = []
    labels = []

    # Minimized test suites
    for preset, name in zip(PRESETS, PRESET_NAMES):
        labels.append(name)
        values.append(
            [
                sum(
                    [
                        num_cases_per_test[t]
                        for t in minimized_test_suites[(preset, project)].tests_minimized_by_mutation_score[0]
                    ]
                )
                for project in PROJECTS
            ]
        )

    labels.append("Pynguin (individual)")
    values.append([len(s.tests_minimized_by_mutation_score.tests) for s in minimized_pynguin_test_suites.values()])

    labels.append("Pynguin (combined)")
    values.append(
        [len(s.tests_minimized_by_mutation_score.tests) for s in minimized_combined_pynguin_test_suites.values()]
    )

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Number of tests")
    distribution_plot(values, ax=ax)

    mannwhitneyu_test(values, labels, "mannwhitneyu_plot_number_of_minimized_test_cases_distplot.csv")
    return fig, {key: value for key, value in zip(labels, [sum(val) / len(val) for val in values])}


# %% Plot LOC in minimized test suites


@block
@savefig
def plot_loc_of_minimized_test_suites_distplot():
    values = []
    labels = []

    # Minimized test suites
    for preset, name in zip(PRESETS, PRESET_NAMES):
        labels.append(name)
        values.append(
            [
                sum(
                    [
                        loc_per_test[t]
                        for t in minimized_test_suites[(preset, project)].tests_minimized_by_mutation_score[0]
                    ]
                )
                for project in PROJECTS
            ]
        )

    labels.append("Pynguin (individual)")
    pynguin_values = []
    for project in PROJECTS:
        for index in range(1, 31):
            if is_pynguin_run_excluded(project, index):
                continue

            project_loc = 0
            for key, loc in pynguin_loc_per_test_minimized_individual.items():
                if key.startswith(f"{index:02}::{project}"):
                    project_loc += loc

            pynguin_values.append(project_loc)
    values.append(pynguin_values)

    labels.append("Pynguin (combined)")
    pynguin_values = []
    for project in PROJECTS:
        project_loc = 0
        for key, loc in pynguin_loc_per_test_minimized_combined.items():
            if re.match(rf"\d\d::{project}", key):
                project_loc += loc
        # print(f"{project}, {project_loc}")
        pynguin_values.append(project_loc)
    values.append(pynguin_values)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Lines of code")
    distribution_plot(values, ax=ax)

    mannwhitneyu_test(values, labels, "mannwhitneyu_plot_loc_of_minimized_test_suites_distplot.csv")
    return fig, {key: value for key, value in zip(labels, [sum(val) / len(val) for val in values])}


# %% Number of Iterations
# ======================================================================================================================


class ____Number_of_Iterations:  # mark this in the outline
    pass


stats_per_iteration_limit = {}


class Usage(NamedTuple):
    cached_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cost: float


class IterationLimitInfo(NamedTuple):
    iterations: int
    successful_runs: List[MutantId]
    kills: Set[MutantId]
    usages: List[Usage]
    # covered_lines: List[str]
    # covered_branches: List[str]


def get_usage_for_n_iterations(conversation, n) -> Usage:
    cached_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    i = 0
    for msg in conversation:
        if i == n:
            break
        if not msg.get("usage"):
            continue

        cached_tokens += msg["usage"].get("cached_tokens", 0)
        prompt_tokens += msg["usage"]["prompt_tokens"]
        completion_tokens += msg["usage"]["completion_tokens"]

        if msg["tag"] in ["experiment_stated", "test_stated"]:
            i += 1

    return Usage(
        cached_tokens,
        prompt_tokens,
        completion_tokens,
        (prompt_tokens - cached_tokens) * 0.150 / 1_000_000
        + cached_tokens * 0.075 / 1_000_000
        + completion_tokens * 0.600 / 1_000_000,
    )


def get_total_usages_for_n_iterations(group, n) -> List[Usage]:
    return list(group["conversation"].map(lambda c: get_usage_for_n_iterations(c, n)))


@block
def calculate_number_of_kills_per_iteration_limit():
    grouped_data = data.groupby("preset")

    for preset in PRESETS:
        # print(f"\n{preset}")
        group = grouped_data.get_group(preset)
        values = []

        for num_turns in range(0, 11):
            group_killed = group[group["mutant_killed"] & (group["num_turns"] <= num_turns)]
            successful_runs = list(group_killed["mutant_id"])
            kills = get_kills(group_killed, group_killed["escaped_long_id"])
            usages = get_total_usages_for_n_iterations(group, num_turns)
            # if we ignore the successful runs, we get exponential growth
            # seems like the fact that some runs end at each iteration counteract this
            # usages = get_total_usages_for_n_iterations(group[~group["mutant_killed"]], num_turns)
            # print(f"number of test files for {num_turns} its: {len(successful_runs)}")
            # print(f"number killed mutants for {num_turns} its: {len(kills)}")
            # print(f"cost for {num_turns} its: {sum([u.cost for u in usages])}")
            values.append(IterationLimitInfo(num_turns, successful_runs, kills, usages))

        stats_per_iteration_limit[preset] = values


# %% Plot Number of Iterations


@block
@savefig
def plot_number_of_kills_per_iteration_limit():
    fig, ax = plt.subplots(layout="constrained", figsize=(6, 4))
    ax.set_xlim(1, 10)
    ax2 = ax.twinx()

    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(0.7)
        ax2.spines[axis].set_linewidth(0.7)

    for preset, name, color in zip(PRESETS[1:], PRESET_NAMES[1:], cmap_colors[::2]):
        info = stats_per_iteration_limit[preset]
        if preset == "baseline_without_iterations":
            continue

        ax.plot(
            [i.iterations for i in info],
            [len(i.kills) for i in info],
            label=f"{name}",
            color=color,
        )
        ax2.plot(
            [i.iterations for i in info],
            [sum([u.cost for u in i.usages]) for i in info],
            "--",
            label=f"{name}",
            color=color,
        )

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles + [Line2D([], [], linestyle="--", color="gray")],
        labels + ["Cost in US-$"],
    )
    ax.set_ylabel("Number of killed mutants")
    ax2.set_ylabel("Cost in US-$")
    ax.set_xlabel("Number of allowed iterations")
    ax.set_xticks(list(range(1, 11)))

    return (
        fig,
        {
            preset: dict(kills=kills, costs=costs)
            for preset, kills, costs in zip(
                PRESET_NAMES,
                [[len(info.kills) for info in stats_per_iteration_limit[preset]] for preset in PRESETS],
                [
                    [sum([u.cost for u in info.usages]) for info in stats_per_iteration_limit[preset]]
                    for preset in PRESETS
                ],
            )
        },
    )


# %% Plot Cost per Kill for Number of Iterations


@block
@savefig
def plot_cost_per_kill_per_iteration_limit():
    fig, ax = plt.subplots(layout="constrained", figsize=(5.75, 4))
    ax.set_xlim(1, 10)

    for preset, name, color in zip(PRESETS[1:], PRESET_NAMES[1:], cmap_colors[::2]):
        info = stats_per_iteration_limit[preset]
        if preset == "baseline_without_iterations":
            continue
        elif preset == "baseline_with_iterations":
            info = info[1:]
        elif preset.startswith("debugging"):
            info = info[2:]

        ax.plot(
            [i.iterations for i in info],
            [sum([u.cost for u in i.usages]) / len(i.kills) if i.kills else 0 for i in info],
            label=f"{name}",
            color=color,
        )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x * 100:.2f}¢")
    ax.set_ylabel("Cost per killed mutant")
    ax.set_xlabel("Number of allowed iterations")
    ax.set_xticks(list(range(1, 11)))

    return (
        fig,
        {
            preset: cost_per_kill
            for preset, cost_per_kill in zip(
                PRESET_NAMES,
                [
                    [
                        sum([u.cost for u in i.usages]) / len(i.kills) if i.kills else 0
                        for i in stats_per_iteration_limit[preset]
                    ]
                    for preset in PRESETS
                ],
            )
        },
    )


# %% Other Stuff
# ======================================================================================================================


class ____Other_Stuff:  # mark this in the outline
    pass


# %% Plot killed mutants by number of claims


@block
@savefig
def plot_killed_mutants_by_num_claims():
    plot_data = data[data["preset"] == PRESETS[0]].groupby("mutant.num_equivalence_claims")

    num_killed_in_conversation = []
    num_killed_cosmic_ray = []
    num_unkilled = []

    for i in [0, 1, 2, 3]:
        group = plot_data.get_group(i)
        claims_killed_other_conversation = group[(group["mutant.num_kills"] > 0)]
        claims_killed_cosmic_ray = group[(group["mutant.num_kills"] == 0) & group["cosmic_ray_full.killed_by_any"]]
        claims_unkilled = group[(group["mutant.num_kills"] == 0) & ~group["cosmic_ray_full.killed_by_any"]]
        num_killed_in_conversation.append(len(claims_killed_other_conversation))
        num_killed_cosmic_ray.append(len(claims_killed_cosmic_ray))
        num_unkilled.append(len(claims_unkilled))

    num_killed_in_conversation = np.array(num_killed_in_conversation)
    num_killed_cosmic_ray = np.array(num_killed_cosmic_ray)
    num_unkilled = np.array(num_unkilled)

    fig, ax = plt.subplots(layout="constrained", figsize=(3.5, 6.5))
    x = np.arange(len(num_unkilled))
    ax.bar(x, num_killed_in_conversation, color=cmap_colors[0])
    ax.bar(
        x,
        num_killed_cosmic_ray,
        color=cmap_colors[2],
        bottom=num_killed_in_conversation,
    )
    ax.bar(
        x,
        num_unkilled,
        color=cmap_colors[4],
        bottom=num_killed_in_conversation + +num_killed_cosmic_ray,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["0", "1", "2", "3"], rotation=90)
    ax.set_ylabel("Number of mutants")
    ax.set_xlabel("Number equivalence claims")
    fig.legend(
        [
            "Killed in con-\nversation",
            "Killed by any\ngenerated test",
            "Not killed",
        ],
        loc=(0.425, 0.8),
    )
    return fig


# %% Check kill rate of claimed and unclaimed mutants

# Not a good idea, I think. Construct validity?
#   LLM claims mutants that it struggles with => LLM struggles with the mutants it claimed


@block
def check_kill_rate_of_claimed_and_unclaimed_mutants():
    grouped_data = data.groupby("preset")

    for preset in PRESETS:
        group = grouped_data.get_group(preset)
        group = group[~group["mutant_killed"]]

        print(preset)
        claimed = group[group["claimed_equivalent"]]["cosmic_ray_full.killed_by_any"]
        not_claimed = group[~group["claimed_equivalent"]]["cosmic_ray_full.killed_by_any"]
        print(f"claimed equivalent: {claimed.sum():4d} killed, {len(claimed) - claimed.sum():4d} alive")
        print(f"       not claimed: {not_claimed.sum():4d} killed, {len(not_claimed) - not_claimed.sum():4d} alive")
        print(mannwhitneyu(claimed, not_claimed))
        print()


# claimed = data[data["claimed_equivalent"]]["cosmic_ray_full.killed_by_own"]
# not_claimed = data[~data["claimed_equivalent"]]["cosmic_ray_full.killed_by_own"]
# print(f"claimed: {claimed.sum()} killed, {len(claimed) - claimed.sum()} alive")
# print(f"not claimed: {not_claimed.sum()} killed, {len(not_claimed) - not_claimed.sum()} alive")
# print(mannwhitneyu(claimed, not_claimed))
# print()


# %% Print mean number of turns it takes to output a successful test


@block
def print_mean_num_turns_for_successful_test():
    print(
        "Baseline: Avg. number of turns for a successful test:",
        data[(data["preset"] == "baseline_with_iterations") & (data["mutant_killed"])]["num_turns"].mean(),
    )

    print(
        "Scientific: Avg. number of turns for a successful test:",
        data[(data["preset"].isin(["debugging_one_shot", "debugging_zero_shot"])) & (data["mutant_killed"])][
            "num_turns"
        ].mean(),
    )

    print(data[data["mutant_killed"]].groupby("preset")["num_turns"].mean())


# %% Plot equivalences per project


@block
@savefig
def plot_equivalences_by_project():
    grouped_data = data[data["preset"] == PRESETS[0]].groupby("project")

    num_mutants_claimed_unkilled = []
    num_mutants_claimed_cosmic_killed = []

    for project in PROJECTS:
        group = grouped_data.get_group(project)
        num_mutants_claimed_unkilled.append(
            len(group[(group["mutant.num_equivalence_claims"] > 0) & (~group["cosmic_ray_full.killed_by_any"])])
            / len(group)
        )
        num_mutants_claimed_cosmic_killed.append(
            len(group[(group["mutant.num_equivalence_claims"] > 0) & (group["cosmic_ray_full.killed_by_any"])])
            / len(group)
        )

    num_mutants_claimed_unkilled = np.array(num_mutants_claimed_unkilled)
    num_mutants_claimed_cosmic_killed = np.array(num_mutants_claimed_cosmic_killed)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.bar(
        x=np.arange(len(PROJECTS)),
        height=num_mutants_claimed_unkilled,
        color=cmap_colors[0],
    )
    ax.bar(
        x=np.arange(len(PROJECTS)),
        bottom=num_mutants_claimed_unkilled,
        height=num_mutants_claimed_cosmic_killed,
        color=cmap_colors[1],
    )

    ax.set_xticks(np.arange(len(PROJECTS)))
    ax.set_xticklabels(PROJECTS, rotation=90)
    ax.yaxis.set_major_formatter(format_perecent())
    labels = [
        "Flagged mutants (unkilled)",
        "Flagged mutants (killed)",
    ]
    fig.legend(labels, loc="upper right", framealpha=1, bbox_to_anchor=(1.05, 1.02))
    ax.set_ylabel("Percentage of mutants")

    plot_data = {}
    for label, project_values in zip(labels, [num_mutants_claimed_unkilled, num_mutants_claimed_cosmic_killed]):
        plot_data[label] = {key: value for key, value in zip(PROJECTS, project_values)}
    return fig, plot_data


# %% Plot equivalences per project


@block
@savefig
def plot_num_claims_by_project():
    grouped_data = data[data["preset"] == PRESETS[0]].groupby("project")

    num_one_claim = []
    num_two_claims = []
    num_three_claims = []

    for project in PROJECTS:
        group = grouped_data.get_group(project)
        num_one_claim.append(len(group[(group["mutant.num_equivalence_claims"] == 1)]))
        num_two_claims.append(len(group[(group["mutant.num_equivalence_claims"] == 2)]))
        num_three_claims.append(len(group[(group["mutant.num_equivalence_claims"] == 3)]))

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.bar(
        x=np.arange(len(PROJECTS)),
        height=num_one_claim,
        color=cmap_colors[0],
    )
    ax.bar(
        x=np.arange(len(PROJECTS)),
        height=num_two_claims,
        bottom=num_one_claim,
        color=cmap_colors[1],
    )
    ax.bar(
        x=np.arange(len(PROJECTS)),
        height=num_three_claims,
        bottom=[a + b for a, b in zip(num_one_claim, num_two_claims)],
        color=cmap_colors[2],
    )

    ax.set_xticks(np.arange(len(PROJECTS)))
    ax.set_xticklabels(PROJECTS, rotation=90)
    ax.yaxis.set_major_formatter(format_perecent())
    labels = [
        "Flagged mutants (one claim)",
        "Flagged mutants (two claims)",
        "Flagged mutants (three claims)",
    ]
    fig.legend(labels, loc="upper right", framealpha=1, bbox_to_anchor=(1.05, 1.02))
    ax.set_ylabel("Percentage of mutants")

    plot_data = {}
    for label, project_values in zip(labels, [num_one_claim, num_two_claims, num_three_claims]):
        plot_data[label] = {key: value for key, value in zip(PROJECTS, project_values)}
    return fig, plot_data


# %% Equivalence claims per project


@block
@savefig
def plot_equivalences_by_project_with_direct_kills():
    grouped_data = data[data["preset"] == PRESETS[0]].groupby("project")

    num_mutants_claimed_unkilled = []
    num_mutants_claimed_direct_killed = []
    num_mutants_claimed_cosmic_killed = []

    for project in PROJECTS:
        group = grouped_data.get_group(project)
        num_mutants_claimed_unkilled.append(
            len(
                group[
                    (group["mutant.num_equivalence_claims"] > 0)
                    & (group["mutant.num_kills"] == 0)
                    & (~group["cosmic_ray_full.killed_by_any"])
                ]
            )
            / len(group)
        )
        num_mutants_claimed_direct_killed.append(
            len(group[(group["mutant.num_equivalence_claims"] > 0) & (group["mutant.num_kills"] > 0)]) / len(group)
        )
        num_mutants_claimed_cosmic_killed.append(
            len(
                group[
                    (group["mutant.num_equivalence_claims"] > 0)
                    & (group["mutant.num_kills"] == 0)
                    & (group["cosmic_ray_full.killed_by_any"])
                ]
            )
            / len(group)
        )

    num_mutants_claimed_unkilled = np.array(num_mutants_claimed_unkilled)
    num_mutants_claimed_direct_killed = np.array(num_mutants_claimed_direct_killed)
    num_mutants_claimed_cosmic_killed = np.array(num_mutants_claimed_cosmic_killed)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 6))
    ax.bar(
        x=np.arange(len(PROJECTS)),
        height=num_mutants_claimed_unkilled,
        color=cmap_colors[0],
    )
    ax.bar(
        x=np.arange(len(PROJECTS)),
        bottom=num_mutants_claimed_unkilled,
        height=num_mutants_claimed_direct_killed,
        color=mix_colors(cmap_colors[0], cmap_colors[1], 2),
    )
    ax.bar(
        x=np.arange(len(PROJECTS)),
        bottom=num_mutants_claimed_unkilled + num_mutants_claimed_direct_killed,
        height=num_mutants_claimed_cosmic_killed,
        color=cmap_colors[1],
    )

    ax.set_xticks(np.arange(len(PROJECTS)))
    ax.set_xticklabels(PROJECTS, rotation=90)
    ax.yaxis.set_major_formatter(format_perecent())
    labels = [
        "Flagged mutants (unkilled)",
        "Flagged mutants (killed when targeted)",
        "Flagged mutants (killed by any test)",
    ]
    fig.legend(labels, loc="lower left", framealpha=1, bbox_to_anchor=(0, -0.15))
    ax.set_ylabel("Percentage of mutants")

    plot_data = {}
    for label, project_values in zip(
        labels, [num_mutants_claimed_unkilled, num_mutants_claimed_direct_killed, num_mutants_claimed_cosmic_killed]
    ):
        plot_data[label] = {key: value for key, value in zip(PROJECTS, project_values)}
    return fig, plot_data


# %% All flagged mutants

data[(data["preset"] == PRESETS[0]) & (data["mutant.num_equivalence_claims"] > 0)]["mutant_id"].count()


# %% Flagged mutants not directly killed

data[(data["preset"] == PRESETS[0]) & (data["mutant.num_equivalence_claims"] > 0) & (data["mutant.num_kills"] == 0)][
    "mutant_id"
].count()


# %% Flagged mutants unkilled

data[
    (data["preset"] == PRESETS[0])
    & (data["mutant.num_equivalence_claims"] > 0)
    & (~data["cosmic_ray_full.killed_by_any"])
]["mutant_id"].count()


# %% Number of killed mutants by test

most_failing_tests = {}

# for index, row in data[(data["mutant.num_equivalence_claims"] > 0)].iterrows():
for index, row in data.iterrows():
    for test_id in row["cosmic_ray_full.failing_tests"]:
        most_failing_tests[test_id] = most_failing_tests.get(test_id, 0) + 1

for entry in sorted(list(most_failing_tests.items()), key=lambda x: x[1]):
    print(entry)


# %% Snakey diagram for flagged, killed and sampled mutants

# Mutants killed in runs but not by cosmic-ray can be explained by flaky tests we had to exclude (and one ineffective test: 84783f0f).
# Mutants killed by cosmic-ray but sampled can be explained by sampling the mutants with old cosmic-ray data.


@block
def samples_snakey():
    plot_data = data[data["preset"] == PRESETS[0]]

    cond_flagged = plot_data["mutant.num_equivalence_claims"] > 0
    cond_directly_killed = plot_data["mutant.num_kills"] > 0
    cond_cosmic_ray_killed = plot_data["cosmic_ray_full.killed_by_any"]
    cond_sampled = plot_data["sampled"]
    cond_equivalent = plot_data["sampled"] & plot_data["sample.equivalent"].fillna(False)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=[
                        "Mutants flagged in a run",  # 0
                        "Mutants not flagged in a run",  # 1
                        "Mutants killed in a run",  # 2
                        "Mutants not killed in a run",  # 3
                        "Mutants killed by any LLM-generated test",  # 4
                        "Mutants not killed by any LLM-generated test",  # 5
                        "Sampled Mutants",  # 6
                        "Equivalent by our definition",  # 7
                        "Not equivalent by our definition",  # 8
                    ],
                    align="right",
                ),
                link=dict(
                    source=[0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6],
                    target=[2, 3, 2, 3, 4, 5, 4, 5, 6, 6, 7, 8],
                    value=[
                        len(plot_data[cond_flagged & cond_directly_killed]),
                        len(plot_data[cond_flagged & ~cond_directly_killed]),
                        len(plot_data[~cond_flagged & cond_directly_killed]),
                        len(plot_data[~cond_flagged & ~cond_directly_killed]),
                        len(plot_data[cond_directly_killed & cond_cosmic_ray_killed]),
                        len(plot_data[cond_directly_killed & ~cond_cosmic_ray_killed]),
                        len(plot_data[~cond_directly_killed & cond_cosmic_ray_killed]),
                        len(plot_data[~cond_directly_killed & ~cond_cosmic_ray_killed]),
                        len(plot_data[cond_cosmic_ray_killed & cond_sampled]),
                        len(plot_data[~cond_cosmic_ray_killed & cond_sampled]),
                        len(plot_data[cond_sampled & cond_equivalent]),
                        len(plot_data[cond_sampled & ~cond_equivalent]),
                    ],
                ),
            )
        ]
    )

    for line in [
        f"len(plot_data),                                                   {len(plot_data)}",
        f"len(plot_data[cond_flagged]),                                     {len(plot_data[cond_flagged])}",
        f"len(plot_data[~cond_flagged]),                                    {len(plot_data[~cond_flagged])}",
        f"len(plot_data[cond_directly_killed]),                             {len(plot_data[cond_directly_killed])}",
        f"len(plot_data[~cond_directly_killed]),                            {len(plot_data[~cond_directly_killed])}",
        f"len(plot_data[cond_cosmic_ray_killed]),                           {len(plot_data[cond_cosmic_ray_killed])}",
        f"len(plot_data[~cond_cosmic_ray_killed]),                          {len(plot_data[~cond_cosmic_ray_killed])}",
        f"len(plot_data[cond_sampled]),                                     {len(plot_data[cond_sampled])}",
        f"len(plot_data[~cond_sampled]),                                    {len(plot_data[~cond_sampled])}",
        f"len(plot_data[cond_equivalent]),                                  {len(plot_data[cond_equivalent])}",
        f"len(plot_data[~cond_equivalent]),                                 {len(plot_data[~cond_equivalent])}",
        "",
        f"len(plot_data[cond_flagged & cond_directly_killed]),              {len(plot_data[cond_flagged & cond_directly_killed])}",
        f"len(plot_data[cond_flagged & ~cond_directly_killed]),             {len(plot_data[cond_flagged & ~cond_directly_killed])}",
        f"len(plot_data[~cond_flagged & cond_directly_killed]),             {len(plot_data[~cond_flagged & cond_directly_killed])}",
        f"len(plot_data[~cond_flagged & ~cond_directly_killed]),            {len(plot_data[~cond_flagged & ~cond_directly_killed])}",
        f"len(plot_data[cond_directly_killed & cond_cosmic_ray_killed]),    {len(plot_data[cond_directly_killed & cond_cosmic_ray_killed])}",
        f"len(plot_data[cond_directly_killed & ~cond_cosmic_ray_killed]),   {len(plot_data[cond_directly_killed & ~cond_cosmic_ray_killed])}",
        f"len(plot_data[~cond_directly_killed & cond_cosmic_ray_killed]),   {len(plot_data[~cond_directly_killed & cond_cosmic_ray_killed])}",
        f"len(plot_data[~cond_directly_killed & ~cond_cosmic_ray_killed]),  {len(plot_data[~cond_directly_killed & ~cond_cosmic_ray_killed])}",
        f"len(plot_data[cond_cosmic_ray_killed & cond_sampled]),            {len(plot_data[cond_cosmic_ray_killed & cond_sampled])}",
        f"len(plot_data[~cond_cosmic_ray_killed & cond_sampled]),           {len(plot_data[~cond_cosmic_ray_killed & cond_sampled])}",
        f"len(plot_data[cond_sampled & cond_equivalent]),                   {len(plot_data[cond_sampled & cond_equivalent])}",
        f"len(plot_data[cond_sampled & ~cond_equivalent]),                  {len(plot_data[cond_sampled & ~cond_equivalent])}",
        "",
        f"len(plot_data[cond_flagged & cond_cosmic_ray_killed]),    {len(plot_data[cond_flagged & cond_cosmic_ray_killed])}",
        f"len(plot_data[cond_flagged & ~cond_cosmic_ray_killed]),   {len(plot_data[cond_flagged & ~cond_cosmic_ray_killed])}",
        f"len(plot_data[~cond_flagged & cond_cosmic_ray_killed]),   {len(plot_data[~cond_flagged & cond_cosmic_ray_killed])}",
        f"len(plot_data[~cond_flagged & ~cond_cosmic_ray_killed]),  {len(plot_data[~cond_flagged & ~cond_cosmic_ray_killed])}",
        "",
        f"len(plot_data[cond_flagged & ~cond_directly_killed & ~cond_cosmic_ray_killed)]),   {len(plot_data[cond_flagged & ~cond_directly_killed & ~cond_cosmic_ray_killed])}",
    ]:
        print(line)

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    fig.write_html(OUTPUT_PATH / "snakey.html")


# https://sankeymatic.com/build/

# Mutants [2427] Flagged
# Mutants [4953] Not Flagged
# Flagged [1673] Killed
# Flagged [754] Unkilled
# Not Flagged [4675] Killed
# Not Flagged [278] Unkilled
# Killed [4] Sampled
# Unkilled [161] Sampled
# Sampled [29] Equivalent
# Sampled [136] Not Equivalent


# %% Mutants sampled but killed by cosmic ray

# (Mutants were sampled with old cosmic-ray data)

data[(data["preset"] == PRESETS[0]) & data["cosmic_ray_full.killed_by_any"] & data["sampled"]]["mutant_id"]


# %% Mutants killed in a run but not killed by cosmic ray

# (84783f0f was found to be an ineffective test, the others were all found to be flaky.)

data[(data["mutant_killed"]) & ~data["cosmic_ray_full.killed_by_any"]]["long_id"]

data[data["num_final_test_cases"] == 3]

# %% Count timeouts


@block
def count_cosmic_ray_timeouts():
    timeouts = 0
    runs = 0
    # Maps (project, preset) to a map that maps (target_path, mutant_op, occurrence) to mutant test result
    for mutants_path in RESULTS_DIR.glob("*/cosmic-ray*/mutants.sqlite"):
        results_dir = (mutants_path / ".." / "..").resolve()
        id = LongId.parse_quoted(results_dir.name)
        results = read_mutant_results(results_dir / "cosmic-ray-full" / "mutants.sqlite", id.project)
        for result in results.values():
            if "Timeout" in result.output:
                timeouts += 1
            runs += 1

    print(f"{timeouts} timeouts out of {runs} runs")


# %% Save minimized pynguin test suites


@block
def dump_minimized_pynguin_tests_individual():
    tests = {}
    for project in PROJECTS:
        tests_per_project = {}

        for index in range(1, 31):
            mini = minimized_pynguin_test_suites.get((project, index))
            if mini is not None:
                tests_per_project[index] = list(mini.tests_minimized_by_mutation_score.tests)

        tests[project] = tests_per_project
    with (OUTPUT_PATH / "pynguin_minimized_tests_individual.json").open("w") as f:
        json.dump(tests, f)


@block
def dump_minimized_pynguin_tests_combined():
    tests = {}
    for project in PROJECTS:
        tests_per_project = {}

        mini = minimized_combined_pynguin_test_suites.get(project)
        if mini is None:
            continue

        for index in range(1, 31):
            tests_per_project[index] = [
                name for name in mini.tests_minimized_by_mutation_score.tests if int(name.split("::")[0]) == index
            ]
        tests[project] = tests_per_project

    with (OUTPUT_PATH / "pynguin_minimized_tests_combined.json").open("w") as f:
        json.dump(tests, f)


# %%

pynguin_data[pynguin_data["failing_tests"].map(len) != 0].head()


# %%
for pre, pro in itertools.product(PRESETS, PROJECTS):
    num = sum([loc_per_test[t] for t in minimized_test_suites[(pre, pro)].tests_minimized_by_mutation_score[0]])
    print(f"{pre} {pro} {num}")

# %% Compute the cost that prompt caching saved


@block
def compute_prompt_caching_saves():
    cost = data["usage.cost"].sum()
    cost_without_caching = (
        data["usage.uncached_prompt_tokens"].sum() * 0.150 / 1_000_000
        + data["usage.cached_tokens"].sum() * 0.150 / 1_000_000
        + data["usage.completion_tokens"].sum() * 0.600 / 1_000_000
    )
    print(f"Prompt caching saved {1 - cost / cost_without_caching}")


# %% Compute the cost of completion tokens


@block
def compute_completion_tokens_cost():
    cost = data["usage.cost"].sum()
    cost_completion = data["usage.cost.completion_tokens"].sum()
    print(f"Completion tokens made up {cost_completion / cost} of cost")


# %% Compute the cost of completion tokens


@block
def compute_completion_tokens_cost():
    cost = data["usage.total_tokens"].sum()
    cost_completion = data["usage.completion_tokens"].sum()
    print(f"Completion tokens made up {cost_completion / cost} of cost")


# %% Compute the cost of completion tokens


@block
def compute_cached_tokens_per_approach():
    grouped_data = data.groupby("preset")
    for preset in PRESETS:
        group = grouped_data.get_group(preset)
        prompt_tokens = group["usage.cached_tokens"].sum() + group["usage.uncached_prompt_tokens"].sum()
        cached_tokens = group["usage.cached_tokens"].sum()
        print(f"{preset}: {cached_tokens / prompt_tokens} of tokens are cached")


# %%

print(len(data[data["mutant_killed"] & data["claimed_equivalent"]]))
print(len(data))


def get_nth_message(n):
    def retv(c):
        try:
            return [msg for msg in c if msg["role"] == "assistant"][n]["content"]
        except IndexError:
            return ""

    return retv


(data["conversation"].map(get_nth_message(0)).map(lambda m: bool(re.match(EQUIVALENCE_HEADLINE_REGEX, m))).sum())

# %%


@block
def difference_in_turns_to_successful_tests():
    iter = data[data["mutant_killed"] & (data["preset"] == "baseline_with_iterations")]["num_turns"].mean()
    scientific = data[data["mutant_killed"] & (data["preset"].map(lambda s: s.startswith("debugging")))][
        "num_turns"
    ].mean()
    print(scientific - iter)


@block
def difference_in_turns_to_successful_tests_other():
    iter = data[data["mutant_killed"] & (data["preset"] == "baseline_with_iterations")]["num_turns"].mean()
    zero_s = data[data["mutant_killed"] & (data["preset"] == "debugging_zero_shot")]["num_turns"].mean()
    one_s = data[data["mutant_killed"] & (data["preset"] == "debugging_one_shot")]["num_turns"].mean()
    print((zero_s + one_s) / 2 - iter)


# data[data["outcome"]]
# for preset in PRESETS:
#     print("\n" + preset)
#     for i in range(11):
#         num_turns = len(data[data["claimed_equivalent"] & (data["num_turns_before_equivalence_claim"] == i) & (data["preset"] == preset)])
#         print(f"{i}: {num_turns}")


# %%


@block
def test_threshold():
    iterative_success_data = data[
        data["mutant_killed"] & data["claimed_equivalent"] & (data["preset"] != "baseline_without_iterations")
    ]
    print(len(iterative_success_data[iterative_success_data["num_turns"] > 7]) / len(iterative_success_data))


# %%


@block
def tests_with_most_kills():
    from collections import Counter

    c = Counter()
    for failing_tests in data["cosmic_ray_full.failing_tests"]:
        c.update(failing_tests)
    print(c.most_common(10))


# %%
len(
    data[
        (data["preset"] == PRESETS[0])
        & (data["mutant.num_equivalence_claims"] > 0)
        # & (data["mutant.num_kills"] == 0)
        & (data["cosmic_ray_full.killed_by_any"])
    ]
)
