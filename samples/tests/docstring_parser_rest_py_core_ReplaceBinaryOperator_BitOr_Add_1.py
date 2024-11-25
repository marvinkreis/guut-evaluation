import re
from docstring_parser.rest import parse


def test():
    orig_add = re.RegexFlag.__add__
    def mock_add(self, other):
        assert False
    re.RegexFlag.__add__ = mock_add
    parse("text")
