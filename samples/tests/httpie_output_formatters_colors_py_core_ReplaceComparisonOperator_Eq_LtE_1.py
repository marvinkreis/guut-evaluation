from httpie.output.formatters.colors import ColorFormatter, SimplifiedHTTPLexer
from httpie.context import Environment


def test():
    c = ColorFormatter(
        Environment(colors=2),
        True,
        "aaaa",
        format_options={}
    )
    assert not isinstance(c.http_lexer, SimplifiedHTTPLexer)
