from flake8.style_guide import Decision, DecisionEngine
from flake8.defaults import SELECT, IGNORE
import argparse


def test():
    options = argparse.Namespace(
        select=[],
        ignore=IGNORE,
        extended_default_select=["X1"],
        enable_extensions=[],
        extend_ignore=[],
        disable_noqa=False,
    )

    decider = DecisionEngine(options)
    assert not decider.using_default_select
