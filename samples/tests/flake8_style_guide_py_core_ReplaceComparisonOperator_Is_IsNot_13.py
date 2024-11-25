import argparse
from unittest.mock import Mock, patch
from flake8.style_guide import Decision, DecisionEngine
from flake8.defaults import SELECT, IGNORE


def test():
    options = argparse.Namespace(
        select=[],
        ignore=["E"],
        extended_default_select=["X1"],
        enable_extensions=[],
        extend_ignore=[],
        disable_noqa=False,
    )

    decider = DecisionEngine(options)
    decider.more_specific_decision_for = lambda: None

    try:
        decider.make_decision("E")
    except TypeError:
        assert False
