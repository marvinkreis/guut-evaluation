import sys

from pdir.color import _Color


def test():
    sys.modules["bpython"] = {}
    color = _Color(3, True)
    print(color.wrap_text("test"))
