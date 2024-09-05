import io
import json
import sys
import os
import traceback
from contextlib import redirect_stderr, redirect_stdout
import tempfile
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from multiprocessing import Pipe, Process
from pathlib import Path
from typing import List

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


@dataclass
class TestResult:
    failed: bool
    is_timeout: bool
    is_import: bool
    exception: str
    stdout: str
    stderr: str


# Stolen from pytest
class CaptureIO(io.TextIOWrapper):
    def __init__(self) -> None:
        super().__init__(io.BytesIO(), encoding="UTF-8", newline="", write_through=True)

    def getvalue(self) -> str:
        assert isinstance(self.buffer, io.BytesIO)
        return self.buffer.getvalue().decode("UTF-8")


def create_mock_io():
    out = CaptureIO()
    out.fileno = sys.stdout.fileno
    err = CaptureIO()
    err.fileno = sys.stderr.fileno
    return out, err


def run_test_module(path: Path, conn):
    with tempfile.TemporaryDirectory() as temp:
        os.chdir(temp)

        module_name = path.stem
        loader = SourceFileLoader(module_name, str(path))

        out, err = create_mock_io()
        try:
            with redirect_stdout(out):
                with redirect_stderr(err):
                    module = loader.load_module()
        except Exception:
            conn.send(
                TestResult(
                    failed=True,
                    is_timeout=False,
                    is_import=True,
                    exception=traceback.format_exc().rstrip(),
                    stdout=out.getvalue(),
                    stderr=err.getvalue(),
                )
            )
            return

        for test in [
            module.__dict__[key] for key in module.__dict__ if "test" in key and callable(module.__dict__[key])
        ]:
            out, err = create_mock_io()
            try:
                with redirect_stdout(out):
                    with redirect_stderr(err):
                        test()
            except Exception:
                conn.send(
                    TestResult(
                        failed=True,
                        is_timeout=False,
                        is_import=False,
                        exception=traceback.format_exc().rstrip(),
                        stdout=out.getvalue(),
                        stderr=err.getvalue(),
                    )
                )
                return

        conn.send(
            TestResult(
                failed=False,
                is_timeout=False,
                is_import=False,
                exception="",
                stdout="",
                stderr="",
            )
        )


def print_fail(module_name: str, result: TestResult):
    print()
    print("-" * 80)

    if result.is_import:
        print(f"{COLOR_RED}IMPORT FAILED: [{module_name}]{COLOR_END}")
    else:
        print(f"{COLOR_RED}TEST FAILED: [{module_name}]{COLOR_END}")

    if result.is_timeout:
        print("-" * 80)
        print("Timeout")
        print("-" * 80)
    else:
        print("-" * 80)
        print("Exception:")
        print(result.exception)
        print("-" * 80)
        print("Stdout:")
        print(result.stdout)
        print("-" * 80)
        print("Stderr:")
        print(result.stderr)
        print("-" * 80)


def run_tests(tests: List[Path]):
    for test_path in tests:
        module_name = test_path.stem

        if module_name in excluded_tests:
            print(f"{COLOR_YELLOW}s{COLOR_END}", end="")
            continue

        parent_conn, child_conn = Pipe()
        process = Process(target=run_test_module, args=(test_path, child_conn))
        process.start()
        process.join(5)

        if process.is_alive():
            process.terminate()
            if process.is_alive():
                process.kill()
            test_result = TestResult(failed=True, is_timeout=True, is_import=False, exception="", stdout="", stderr="")
        else:
            if parent_conn.poll():
                test_result: TestResult = parent_conn.recv()
            else:
                test_result = TestResult(
                    failed=True, is_timeout=True, is_import=False, exception="", stdout="", stderr=""
                )

        if test_result.failed:
            print_fail(module_name, test_result)
            if baseline:
                excluded_tests.add(module_name)
                exclude_file.write_text(json.dumps(list(excluded_tests)))
            else:
                if stop_after_first_fail:
                    sys.exit(1)
        else:
            print(".", end="")


# Read arguments
args = sys.argv
command = args[1]  # "baseline" or "test"
exclude_file = Path(args[2])  # path to "exclude.json" file
if args[3] == "-x":
    stop_after_first_fail = True
    tests_args = args[4:]
else:
    stop_after_first_fail = False
    tests_args = args[3:]

tests = [Path(path) for path in tests_args if not path.endswith("__pycache__")]  # paths to test files

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
sys.path.insert(0, str(Path(".").absolute()))

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
