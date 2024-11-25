from docstring_parser.rest import _build_meta
from docstring_parser.common import ParseError

from collections import UserList


class MockList(UserList):
    def __len__(self):
        return 0


def test():
    try:
        _build_meta(MockList(["raises"]), "")
        assert False
    except ParseError:
        pass

