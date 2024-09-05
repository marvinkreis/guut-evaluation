import shutil
import sys
from pathlib import Path
import pandas as pd
import json

pynguin_results_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

pynguin_results = pd.read_csv(pynguin_results_dir / "results.csv")
package_to_project = json.loads((Path(sys.argv[0]).parent / "package_to_project.json").read_text())


def map_to_project(target_module, seen_unknown_modules=[]):
    if project := package_to_project.get(target_module.split(".")[0]):
        return project
    else:
        if target_module not in seen_unknown_modules:
            print(f"Couldn't map target module to project: {target_module}")
            seen_unknown_modules.append(target_module)
        return "unknown"


pynguin_results["Project"] = pynguin_results["TargetModule"].map(map_to_project)

for index, row in pynguin_results.iterrows():
    src_path = pynguin_results_dir / f"{row["RunId"]}"
    dest_path: Path = output_dir / row["Project"] / f"{(row["RandomSeed"] + 1):02d}" / "tests"

    dest_path.mkdir(exist_ok=True, parents=True)
    for test_file in src_path.glob("*.py"):
        shutil.copyfile(test_file, dest_path / Path(test_file).name)
