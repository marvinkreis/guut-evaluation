from apimd.compiler import load_file
from io import StringIO
import logging
import sys


def test():
    class A:
        def __repr__(self):
            raise Exception()

    out = StringIO()
    logging.getLogger().addHandler(logging.StreamHandler(out))
    load_file("", A())
    assert "__repr__" in out.getvalue()
