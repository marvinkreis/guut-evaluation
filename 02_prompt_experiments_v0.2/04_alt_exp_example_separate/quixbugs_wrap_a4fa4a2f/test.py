from wrap import wrap

def test__wrap():
    """The mutant should not return an empty list when given text that fits within the column width."""
    output = wrap("Hello, World!", 15)
    assert len(output) > 0, "wrap must return a non-empty list when text fits within the column width."