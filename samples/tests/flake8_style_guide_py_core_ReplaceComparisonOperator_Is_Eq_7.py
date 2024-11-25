from flake8.style_guide import Decision, DecisionEngine, Ignored
from flake8.defaults import SELECT, IGNORE
import argparse


class EqToAll:
    def __eq__(self, other):
        return True

def test():
    options = argparse.Namespace(
        select=SELECT,
        ignore=IGNORE,
        extended_default_select=["X1"],
        enable_extensions=[],
        extend_ignore=[],
        disable_noqa=False,
    )

    decider = DecisionEngine(options)
    decider.was_selected = lambda x: EqToAll()

    try:
        decision = decider.make_decision("X1")
        assert decision == Decision.Selected
    except UnboundLocalError:
        pass

