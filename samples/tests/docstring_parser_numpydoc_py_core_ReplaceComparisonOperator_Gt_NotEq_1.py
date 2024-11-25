from docstring_parser.numpydoc import ParamSection
import docstring_parser.numpydoc as numpydoc
from collections import UserString
from types import SimpleNamespace


class MockString(UserString):
    pass

def test():
    section = ParamSection("Notes", "notes")
    def mock_search(value):
        assert False
    numpydoc.PARAM_DEFAULT_REGEX = SimpleNamespace(search=mock_search)


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


    print(section._parse_item("", MockString("")))

