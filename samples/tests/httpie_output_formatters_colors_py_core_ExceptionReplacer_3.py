from httpie.output.formatters.colors import get_lexer


def test():
    try:
        get_lexer("applicaschion/jschon", True, "}}")
    except NameError:
        assert False
