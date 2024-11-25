from docstring_parser.google import GoogleParser, Section, SectionType
from docstring_parser.common import ParseError


def test():
    parser = GoogleParser(sections=[Section("Name", "param", 4)])
    try:
        parser._build_meta("a : 1", "Name")
    except ParseError:
        assert False


