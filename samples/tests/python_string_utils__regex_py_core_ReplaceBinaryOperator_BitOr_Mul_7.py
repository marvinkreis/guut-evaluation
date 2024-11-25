from contextlib import redirect_stdout
from io import StringIO


def test():
    out = StringIO()
    err = StringIO()
    with redirect_stdout(out):
        from string_utils._regex import PRETTIFY_RE
    assert not out.getvalue()
