# %% Imports, constants and helpers

from collections import namedtuple
from pathlib import Path
import sqlite3
import json

import pandas as pd
import os


pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

EVALUATION_DIR = next(path for path in Path(os.getcwd()).parents if path.name == "guut-evaluation")
BENCH_DIR = EVALUATION_DIR / "04_new_prompt_v0.4"
SESSION_FILE = EVALUATION_DIR / "emse_projects/mutants_sampled/python-string-utils.sqlite"

# %% Prepare loading mutants

Mutant = namedtuple(
    "MutantResult",
    ["module_path", "operator_name", "occurrence", "job_id"],
)


def query_mutants(session_file: Path):
    con = sqlite3.connect(session_file)
    with con:
        con.row_factory = lambda cursor, row: Mutant(*row)
        cur = con.cursor()
        cur.execute("""
            select module_path, operator_name, occurrence, job_id
            from mutation_specs;
        """)
        return cur.fetchall()


MutantResult = namedtuple(
    "MutantResult",
    ["module_path", "operator_name", "occurrence", "job_id", "killed", "diff", "output"],
)


def query_mutant_results(session_file: Path):
    con = sqlite3.connect(session_file)
    with con:
        con.row_factory = lambda cursor, row: MutantResult(*row)
        cur = con.cursor()
        cur.execute("""
            select module_path, operator_name, occurrence, mutation_specs.job_id,
                   (test_outcome = 'KILLED') as killed, diff, output
            from mutation_specs, work_results
            where mutation_specs.job_id = work_results.job_id;
        """)
        return cur.fetchall()


# %% Load mutants

BASELINE = "baseline"
WITH_EXAMPLE = "WITH_EXAMPLE"
NO_EXAMPLE = "NO_EXAMPLE"


def mutant_id(mutant_or_result):
    return (mutant_or_result.module_path, mutant_or_result.operator_name, mutant_or_result.occurrence)


mutants = query_mutants(SESSION_FILE)
mutants_dict = {mutant_id(m): m for m in mutants}

results_list = {}
results_dict = {}

results_list[BASELINE] = query_mutant_results(
    BENCH_DIR / "promt_v2__baseline_with_iterations__sampled_mutants__string_utils_1679e39a/cosmic_ray/session.sqlite"
)
results_dict[BASELINE] = {mutant_id(m): m for m in results_list[BASELINE]}

results_list[WITH_EXAMPLE] = query_mutant_results(
    BENCH_DIR / "promt_v2__with_example__sampled_mutants__string_utils_cc56dd0/cosmic_ray/session.sqlite"
)
results_dict[WITH_EXAMPLE] = {mutant_id(m): m for m in results_list[WITH_EXAMPLE]}

results_list[NO_EXAMPLE] = query_mutant_results(
    BENCH_DIR / "promt_v2__without_example__sampled_mutants__string_utils_5e5e3ee/cosmic_ray/session.sqlite"
)
results_dict[NO_EXAMPLE] = {mutant_id(m): m for m in results_list[NO_EXAMPLE]}

# %% Get ids of killed mutants

killed_ids = {}
killed_ids[BASELINE] = {mutant_id(m) for m in results_list[BASELINE] if m.killed}
killed_ids[WITH_EXAMPLE] = {mutant_id(m) for m in results_list[WITH_EXAMPLE] if m.killed}
killed_ids[NO_EXAMPLE] = {mutant_id(m) for m in results_list[NO_EXAMPLE] if m.killed}
# %% Helper functions


def list_mutants(mutants):
    for mutant in mutants:
        if hasattr(mutant, "id"):
            print(mutant.id)
        else:
            print(f"{mutant.module_path:<25} {mutant.operator_name:<50} {mutant.occurrence}")


def print_mutant_results(mutant_results):
    divider = "=================================== FAILURES ==================================="
    for mutant in mutant_results:
        print("█" * 90)
        print(f"{mutant.module_path} {mutant.operator_name} {mutant.occurrence}")
        print(mutant.diff)
        print("░" * 90)
        print(mutant.output[mutant.output.index(divider) + len(divider) :])


# %% List mutants only killed by baseline

list_mutants(mutants_dict[id] for id in (killed_ids[BASELINE] - killed_ids[WITH_EXAMPLE] - killed_ids[NO_EXAMPLE]))

# %% List mutants only killed by loop with example

list_mutants(mutants_dict[id] for id in (killed_ids[WITH_EXAMPLE] - killed_ids[NO_EXAMPLE] - killed_ids[BASELINE]))

# %% List mutants only killed by loop without example

list_mutants(mutants_dict[id] for id in (killed_ids[NO_EXAMPLE] - killed_ids[WITH_EXAMPLE] - killed_ids[BASELINE]))

# %% Print mutants only killed by baseline

print_mutant_results(
    results_dict[BASELINE][id] for id in (killed_ids[BASELINE] - killed_ids[NO_EXAMPLE] - killed_ids[WITH_EXAMPLE])
)

# %% Print mutants only killed by loop with example

print_mutant_results(
    results_dict[WITH_EXAMPLE][id] for id in (killed_ids[WITH_EXAMPLE] - killed_ids[BASELINE] - killed_ids[NO_EXAMPLE])
)

# %% Print mutants only killed by loop without example

print_mutant_results(
    results_dict[NO_EXAMPLE][id] for id in (killed_ids[NO_EXAMPLE] - killed_ids[BASELINE] - killed_ids[WITH_EXAMPLE])
)


# %% Load benchmark results

BENCH_DIR = EVALUATION_DIR / "04_new_prompt_v0.4"

BenchResult = namedtuple(
    "BenchResult",
    ["module_path", "operator_name", "occurrence", "id", "killed"],
)


def parse_bench_result(bench_result):
    return BenchResult(
        module_path=bench_result["problem"]["target_path"],
        operator_name=bench_result["problem"]["mutant_op"],
        occurrence=bench_result["problem"]["occurrence"],
        id=bench_result["id"],
        killed=bench_result["mutant_killed"],
    )


def load_bench_results(loop_dir):
    bench_results = []
    for root, dirs, files in Path(loop_dir).walk():
        result_paths = (root / name for name in files if name == "result.json")
        for path in result_paths:
            result = json.loads(path.read_text())
            bench_results.append(parse_bench_result(result))
    return bench_results


# %% Load benchmark results

bench_results_list = {}
bench_results_dict = {}

bench_results_list[BASELINE] = load_bench_results(
    BENCH_DIR / "promt_v2__baseline_with_iterations__sampled_mutants__string_utils_1679e39a/loops"
)
bench_results_dict[BASELINE] = {mutant_id(m): m for m in bench_results_list[BASELINE]}

bench_results_list[WITH_EXAMPLE] = load_bench_results(
    BENCH_DIR / "promt_v2__with_example__sampled_mutants__string_utils_cc56dd0/loops"
)
bench_results_dict[WITH_EXAMPLE] = {mutant_id(m): m for m in bench_results_list[WITH_EXAMPLE]}

bench_results_list[NO_EXAMPLE] = load_bench_results(
    BENCH_DIR / "promt_v2__without_example__sampled_mutants__string_utils_5e5e3ee/loops"
)
bench_results_dict[NO_EXAMPLE] = {mutant_id(m): m for m in bench_results_list[NO_EXAMPLE]}
# %% Get ids of directly killed mutants

bench_killed_ids = {}
bench_killed_ids[BASELINE] = {mutant_id(m) for m in bench_results_list[BASELINE] if m.killed}
bench_killed_ids[WITH_EXAMPLE] = {mutant_id(m) for m in bench_results_list[WITH_EXAMPLE] if m.killed}
bench_killed_ids[NO_EXAMPLE] = {mutant_id(m) for m in bench_results_list[NO_EXAMPLE] if m.killed}

# %% Sanity check

print(
    f"Baseline: cr: {len(killed_ids[BASELINE])}, bench: {len(bench_killed_ids[BASELINE])}, both: {len(killed_ids[BASELINE] & bench_killed_ids[BASELINE])}"
)
print(
    f"With example: cr: {len(killed_ids[WITH_EXAMPLE])}, bench: {len(bench_killed_ids[WITH_EXAMPLE])}, both: {len(killed_ids[WITH_EXAMPLE] & bench_killed_ids[WITH_EXAMPLE])}"
)
print(
    f"Without example: cr: {len(killed_ids[NO_EXAMPLE])}, bench: {len(bench_killed_ids[NO_EXAMPLE])}, both: {len(killed_ids[NO_EXAMPLE] & bench_killed_ids[NO_EXAMPLE])}"
)


# %% Directly killed mutants only killed by baseline

list_mutants(
    bench_results_dict[BASELINE][id]
    for id in bench_killed_ids[BASELINE] & (killed_ids[BASELINE] - killed_ids[WITH_EXAMPLE] - killed_ids[NO_EXAMPLE])
)

# %% Directly killed mutants only killed by loop with example

list_mutants(
    bench_results_dict[WITH_EXAMPLE][id]
    for id in bench_killed_ids[WITH_EXAMPLE]
    & (killed_ids[WITH_EXAMPLE] - killed_ids[BASELINE] - killed_ids[NO_EXAMPLE])
)

# %% Directly killed mutants only killed by loop without example

list_mutants(
    bench_results_dict[NO_EXAMPLE][id]
    for id in bench_killed_ids[NO_EXAMPLE] & (killed_ids[NO_EXAMPLE] - killed_ids[BASELINE] - killed_ids[WITH_EXAMPLE])
)

# %% Print number of killed mutants

print("Killed mutants:")
print(f"  Baseline: {len(killed_ids[BASELINE])}")
print(f"  Loop with example: {len(killed_ids[WITH_EXAMPLE])}")
print(f"  Loop without example: {len(killed_ids[NO_EXAMPLE])}")
print()

print(f"  Intersection: {len(killed_ids[BASELINE] & killed_ids[WITH_EXAMPLE] & killed_ids[NO_EXAMPLE])}")
print(f"  Union: {len(killed_ids[BASELINE] | killed_ids[WITH_EXAMPLE] | killed_ids[NO_EXAMPLE])}")
print()

print(f"  Only by baseline: {len(killed_ids[BASELINE] - killed_ids[WITH_EXAMPLE] - killed_ids[NO_EXAMPLE])}")
print(f"  Only by loop with example: {len(killed_ids[WITH_EXAMPLE] - killed_ids[BASELINE] - killed_ids[NO_EXAMPLE])}")
print(
    f"  Only by loop without example: {len(killed_ids[NO_EXAMPLE] - killed_ids[BASELINE] - killed_ids[WITH_EXAMPLE])}"
)
print()

print(
    f"  Only by baseline and mutant was the target: {len(bench_killed_ids[BASELINE] & (killed_ids[BASELINE] - killed_ids[WITH_EXAMPLE] - killed_ids[NO_EXAMPLE]))}"
)
print(
    f"  Only by loop with example and mutant was the target: {len(bench_killed_ids[WITH_EXAMPLE] & (killed_ids[WITH_EXAMPLE] - killed_ids[BASELINE] - killed_ids[NO_EXAMPLE]))}"
)
print(
    f"  Only by loop without example and mutant was the target: {len(bench_killed_ids[NO_EXAMPLE] & (killed_ids[NO_EXAMPLE] - killed_ids[BASELINE] - killed_ids[WITH_EXAMPLE]))}"
)
