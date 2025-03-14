import concurrent.futures
import io
import json
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import List, cast

if sys.stdout.isatty():
    COLOR_BLUE = "\033[34m"
    COLOR_END = "\033[0m"
else:
    COLOR_BLUE = ""
    COLOR_END = ""


@dataclass
class TestResult:
    failed: bool
    stdout: str
    stderr: str


FAILING_TESTS_REGEX = re.compile(r"FAILED ([^:]+)::([^\s]+)")


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


def patch_test_file(test_path: Path, test_name: str):
    test_src = Path(test_path).read_text()
    if "import pytest" not in test_src:
        test_src = f"import pytest\n{test_src}"
    test_src = test_src.replace(f"def {test_name}", f'@pytest.mark.skip(reason="failed_in_baseline")\ndef {test_name}')
    Path(test_path).write_text(test_src)


def exclude_test(test_path, test_name):
    short_path = Path(test_path).stem
    excluded_tests.add(f"{short_path}::{test_name}")
    exclude_file.write_text(json.dumps(list(excluded_tests)))


def run_tests_inner(tests: List[Path], conn):
    out, err = create_mock_io()
    with redirect_stdout(out):
        with redirect_stderr(err):
            import pytest

            try:
                args = ["--timeout=5", "--timeout-method=signal", "--capture=no"]
                if coverage_arg:
                    args += [coverage_arg]
                exit_code = pytest.main(args + [str(test) for test in tests])
                conn.send(TestResult(failed=(exit_code != 0), stdout=out.getvalue(), stderr=err.getvalue()))
            except Exception:
                conn.send(TestResult(failed=True, stdout=out.getvalue(), stderr=err.getvalue()))


def run_tests(tests: List[Path]):
    event = Event()
    parent_conn, child_conn = Pipe()
    process = Process(target=run_tests_inner, args=(tests, child_conn))
    process.start()

    executor = concurrent.futures.ThreadPoolExecutor()
    future_wait = executor.submit(lambda: process.join(180) or "done")
    future_recv = executor.submit(parent_conn.recv)

    test_result = TestResult(failed=True, stdout="", stderr="")
    for future in concurrent.futures.as_completed([future_wait, future_recv]):
        result: TestResult | str = future.result()
        if result == "done":
            future_recv.cancel()
        else:
            future_wait.cancel()
            test_result = cast(TestResult, result)
            print(test_result.stdout)
            print(test_result.stderr)
        executor.shutdown(wait=False)
        break

    if process.is_alive():
        process.terminate()
        if process.is_alive():
            process.kill()
    executor.shutdown(wait=False)

    if baseline:
        for test_path, test_name in set(re.findall(FAILING_TESTS_REGEX, test_result.stdout)):
            short_path = Path(test_path).stem
            print(f"Test failed: {short_path}::{test_name}")
            exclude_test(test_path, test_name)
            patch_test_file(test_path, test_name)
    else:
        for test_path, test_name in set(re.findall(FAILING_TESTS_REGEX, test_result.stdout)):
            short_path = Path(test_path).stem
            print(f"Test failed: {short_path}::{test_name}")
        sys.exit(1 if test_result.failed else 0)


# Read arguments
argv = sys.argv
coverage_arg = None
for arg in argv:
    if arg.startswith("--cov"):
        coverage_arg = arg
        break
argv = [arg for arg in argv if not arg.startswith("--cov")]

command = argv[1]  # "baseline" or "test"
exclude_file = Path(argv[2])  # path to "exclude.json" file
tests = [Path(path) for path in argv[3:] if not path.endswith("__pycache__")]  # paths to test files

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


new_test_paths = []
with TemporaryDirectory() as tempdir:
    temp_path = Path(tempdir)
    for test_path in tests:
        new_test_path = temp_path / Path(test_path).name
        code = test_path.read_text()
        code = code.replace(".<lambda>", "")  # Discard Cython syntax
        new_test_path.write_text(code)
        new_test_paths.append(new_test_path)

    for excluded_test in excluded_tests:
        short_path, test_name = excluded_test.split("::")
        patch_test_file(temp_path / f"{short_path}.py", test_name)

    # baseline: execute tests 100 times and exclude failing tests
    if command == "baseline":
        baseline = True
        for i in range(100):
            print(f"\n\n{COLOR_BLUE}Running baseline iteration {i+1}{COLOR_END}")
            run_tests(new_test_paths)

    # test: execute tests once
    elif command == "test":
        baseline = False
        print(f"{COLOR_BLUE}Running tests{COLOR_END}")
        run_tests(new_test_paths)
