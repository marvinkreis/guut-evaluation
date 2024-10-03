from string_utils._regex import HTML_RE

def test_html_re_mutant_killing():
    """
    Test the compilation of the HTML_RE regular expression.
    The baseline should compile successfully, while the mutant should raise
    a TypeError due to the misuse of the division operator instead of bitwise
    OR for combining regex flags.
    """
    try:
        html_re = HTML_RE
        assert True, "HTML_RE compiled successfully."
    except TypeError as e:
        assert str(e) == "unsupported operand type(s) for |: 'RegexFlag' and 'float'", f"Unexpected error message: {e}"