# ./run_image.sh run_emse_benchmark debugging-zero-shot flake8

import sys
import json
import shutil
import sqlite3
from pathlib import Path


mutants_file = Path(sys.argv[1])
result_dir = Path(sys.argv[2])
out_path = Path("./patched_mutants.sqlite")


def list_mutants(cur: sqlite3.Cursor):
    cursor.execute("select module_path, operator_name, occurrence from mutation_specs;")
    mutants = [args for args in cursor.fetchall()]
    return mutants


def get_job_ids(cur: sqlite3.Cursor, mutants):
    job_ids = []
    for mutant in mutants:
        cur.execute(
            "select job_id from mutation_specs where module_path = ? and operator_name = ? and occurrence = ?;",
            list(mutant),
        )
        job_ids.append(cursor.fetchone()[0])
    return job_ids


def delete_mutants(cur: sqlite3.Cursor, job_ids):
    cur.executemany(
        "delete from mutation_specs where job_id = ?;",
        [[job_id] for job_id in job_ids],
    )

    cur.executemany(
        "delete from work_items where job_id = ?;",
        [[job_id] for job_id in job_ids],
    )

    cur.executemany(
        "delete from work_results where job_id = ?;",
        [[job_id] for job_id in job_ids],
    )


shutil.copyfile(str(mutants_file), str(out_path))
conn = sqlite3.connect(out_path)
cursor = conn.cursor()


all_mutants = list_mutants(cursor)
done_mutants = []
for results_json_path in result_dir.glob("loops/*/result.json"):
    with results_json_path.open("r") as f:
        results = json.load(f)
        description = results["problem"]
        done_mutants.append(
            (
                description["target_path"],
                description["mutant_op"],
                description["occurrence"],
            )
        )

all_mutants = set(all_mutants)
done_mutants = set(done_mutants)

print(f"Original mutants: {len(all_mutants)}")
print(f"Done mutants: {len(done_mutants)}")
print(f"Remaining mutants: {len(all_mutants - done_mutants)}")

job_ids = get_job_ids(cursor, done_mutants)
delete_mutants(cursor, job_ids)

conn.commit()
conn.execute("vacuum;")
conn.commit()
conn.close()
