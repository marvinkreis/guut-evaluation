import sqlite3
import sys
import random
from pathlib import Path

NUM_MUTANTS = 1000


"""
Samples up to NUM_MUTANTS mutants from a session file.
Usage: ./script session_file [session_file...]
- session_file: mutants session file from cosmic ray
"""


def delete_mutants(cur: sqlite3.Cursor, mutants):
    cur.executemany(
        "delete from mutation_specs where job_id = ?;",
        [m[3:] for m in mutants],
    )

    cur.executemany(
        "delete from work_items where job_id = ?;",
        [m[3:] for m in mutants],
    )

    cur.executemany(
        "delete from work_results where job_id = ?;",
        [m[3:] for m in mutants],
    )


def sample_mutants(session_file: Path):
    con = sqlite3.connect(session_file)
    cur = con.cursor()

    cur.execute(
        "select module_path, operator_name, occurrence, job_id from mutation_specs;"
    )
    mutants = set(cur.fetchall())
    print(f"found {len(mutants)} mutants")

    test_mutants = [m for m in mutants if "test" in m[0]]
    if test_mutants:
        print(f"found {len(test_mutants)} mutants for test code")
        delete_mutants(cur, test_mutants)
        mutants.difference_update(test_mutants)

    if len(mutants) > NUM_MUTANTS:
        sampled_mutants = random.sample(list(mutants), NUM_MUTANTS)
        delete_mutants(cur, mutants.difference(sampled_mutants))

    con.commit()
    con.execute("vacuum;")
    con.commit()
    con.close()


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        sample_mutants(Path(path))
