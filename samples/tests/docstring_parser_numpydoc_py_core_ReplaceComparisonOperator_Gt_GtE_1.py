from docstring_parser.numpydoc import ParamSection
import docstring_parser.numpydoc as numpydoc
from collections import UserString
from types import SimpleNamespace


def test():
    section = ParamSection("Notes", "notes")
    def mock_search(value):
        assert False
    numpydoc.PARAM_DEFAULT_REGEX = SimpleNamespace(seach=mock_search)
    print(section._parse_item("", ""))

