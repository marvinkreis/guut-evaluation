import ast
from unittest.mock import MagicMock, patch
from flake8.plugins.pyflakes import FlakesChecker


def test():
    FlakesChecker.with_doctest = True
    FlakesChecker.include_in_doctest = ["a.py"]
    FlakesChecker.exclude_from_doctest = [""]
    checker = FlakesChecker(ast.parse("import sys"), (), "filename.py")
    assert checker.withDoctest

