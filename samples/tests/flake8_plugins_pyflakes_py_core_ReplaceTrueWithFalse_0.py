import ast
from unittest.mock import MagicMock, patch
from flake8.plugins.pyflakes import FlakesChecker


def test():
    FlakesChecker.with_doctest = False
    FlakesChecker.include_in_doctest = ["filename.py"]
    checker = FlakesChecker(ast.parse("import sys"), (), "filename.py")
    assert checker.withDoctest

