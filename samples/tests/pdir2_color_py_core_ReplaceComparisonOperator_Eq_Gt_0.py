import sys

from pdir.color import _Color


def test():
    sys.modules["bpython"] = {}
    color = _Color(3, True)
    assert len(color.wrap_text("test")) == 16
