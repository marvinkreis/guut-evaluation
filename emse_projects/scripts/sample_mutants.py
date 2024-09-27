import sqlite3
import sys
import random
from pathlib import Path

NUM_MUTANTS = 1000


def sample_mutants(session_file: Path):
    con = sqlite3.connect(session_file)
    cur = con.cursor()

    cur.execute(
        "select module_path, operator_name, occurrence, job_id from mutation_specs;"
    )
    mutants = cur.fetchall()
    print(f"found {len(mutants)} mutants")

    if len(mutants) > NUM_MUTANTS:
        sampled_mutants = random.sample(mutants, NUM_MUTANTS)
        mutant_difference = set(mutants).difference(sampled_mutants)

        cur.executemany(
            "delete from mutation_specs where job_id = ?;",
            [m[3:] for m in mutant_difference],
        )

        cur.executemany(
            "delete from work_items where job_id = ?;",
            [m[3:] for m in mutant_difference],
        )

        cur.executemany(
            "delete from work_results where job_id = ?;",
            [m[3:] for m in mutant_difference],
        )

        con.commit()
        con.execute("vacuum;")
        con.commit()
        con.close()


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        sample_mutants(Path(path))
