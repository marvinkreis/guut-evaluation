from httpie.output.formatters.colors import ColorFormatter, SimplifiedHTTPLexer
from httpie.context import Environment


def test():
    c = ColorFormatter(
        Environment(),
        True,
        "aaaa",
        format_options={}
    )
    assert isinstance(c.http_lexer, SimplifiedHTTPLexer)
