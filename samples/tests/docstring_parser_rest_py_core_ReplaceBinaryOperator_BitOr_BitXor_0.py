import docstring_parser.rest
import sys
from docstring_parser.common import ParseError

def test():
    docstring_parser.rest.YIELDS_KEYWORDS |= docstring_parser.rest.RETURNS_KEYWORDS
    try:
        docstring_parser.rest._build_meta(["return", "b", "c"], "d")
        assert False
    except ParseError:
        pass
