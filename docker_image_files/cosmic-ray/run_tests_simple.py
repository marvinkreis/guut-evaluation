import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Callable, List

if sys.stdout.isatty():
    COLOR_RED = "\033[31m"
    COLOR_YELLOW = "\033[33m"
    COLOR_BLUE = "\033[34m"
    COLOR_END = "\033[0m"
else:
    COLOR_RED = ""
    COLOR_YELLOW = ""
    COLOR_BLUE = ""
    COLOR_END = ""


def run_test(test: Callable, test_id: str, is_import: bool):
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out):
            with redirect_stderr(err):
                retv = test()
        if not is_import:
            print(".", end="")
        return retv
    except Exception:
        print()
        print("-" * 80)
        if is_import:
            print(f"{COLOR_RED}Importing {test_id} failed:{COLOR_END}")
        else:
            print(f"{COLOR_RED}Test {test_id} failed:{COLOR_END}")
        print("-" * 80)
        print(traceback.format_exc().rstrip())
        print("-" * 80)
        print("Stdout:")
        print(out.getvalue())
        print("-" * 80)
        print("Stderr:")
        print(err.getvalue())
        print("-" * 80)

        if baseline:
            excluded_tests.add(test_id)
            exclude_file.write_text(json.dumps(list(excluded_tests)))
        else:
            sys.exit(1)


def run_tests(tests: List[Path]):
    for test_path in tests:
        module_name = test_path.stem
        loader = SourceFileLoader(module_name, str(test_path))

        test_id = module_name
        if test_id in excluded_tests:
            print(f"{COLOR_YELLOW}s{COLOR_END}", end="")
            continue
        module = run_test(loader.load_module, test_id, is_import=True)

        for test in [
            module.__dict__[key] for key in module.__dict__ if "test" in key and callable(module.__dict__[key])
        ]:
            test_id = f"{module_name}:{test.__name__}"
            if test_id in excluded_tests:
                print(f"{COLOR_YELLOW}s{COLOR_END}", end="")
                continue
            run_test(test, test_id, is_import=False)


# Read arguments
command = sys.argv[1]  # "baseline" or "test"
exclude_file = Path(sys.argv[2])  # path to "exclude.json" file
tests = [Path(path) for path in sys.argv[3:] if not path.endswith("__pycache__")]  # paths to test files

if command not in ["baseline", "test"]:
    print(f"Invalid command: {command}", file=sys.stderr)
    sys.exit(1)

# Initialize excluded tests
exclude_file = Path(exclude_file)
if exclude_file.is_file():
    excluded_tests = set(json.loads(exclude_file.read_text()))
else:
    exclude_file.write_text("[]")
    excluded_tests = set()

# Make modules in current dir available to tests
sys.path.append(".")

# baseline: execute tests 100 times and exclude failing tests
if command == "baseline":
    baseline = True
    for i in range(100):
        print(f"\n\n{COLOR_BLUE}Running baseline iteration {i+1}{COLOR_END}")
        run_tests(tests)

# test: execute tests once and fail one a test fails
elif command == "test":
    baseline = False
    print(f"{COLOR_BLUE}Running tests{COLOR_END}")
    run_tests(tests)

# Fix missing newline
print()
