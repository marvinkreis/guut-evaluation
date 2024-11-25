from docstring_parser.numpydoc import RaisesSection
import docstring_parser.numpydoc as numpydoc
from collections import UserString
from types import SimpleNamespace


class MockString(UserString):
    pass

def test():
    section = RaisesSection(MockString("Notes"), "notes")

    orig_len = len
    def patched_len(l):
        try:
            if isinstance(l, MockString):
                __builtins__["len"] = orig_len
                return -1
        except Exception:
            return orig_len(l)
        return orig_len(l)

    __builtins__["len"] = patched_len

    assert section._parse_item(MockString("key"), "value").type_name is None

