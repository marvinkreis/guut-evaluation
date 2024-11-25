from docstring_parser.google import GoogleParser, Section, SectionType
import re

def test():
    sections = [
        Section("Name1", "examples", SectionType.MULTIPLE),
        Section("Name2", "examples", SectionType.MULTIPLE),
        Section("....", "examples", SectionType.MULTIPLE)
    ]
    parser = GoogleParser(sections)
    result = parser.parse(
"""Example function.

Name1:
    param1: The first parameter.

Fake:
    param1: The first parameter.

Name2:
    param1: The first parameter.
""")
    assert len(result.meta) == 2

