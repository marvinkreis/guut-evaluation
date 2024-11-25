from docstring_parser.parser import parse, Style


class A:
    def __eq__(self, other):
        return other is Style.auto

    def __hash__(self):
        return hash(Style.auto)


def test():
    try:
        parse("", A())
    except KeyError:
        assert False

