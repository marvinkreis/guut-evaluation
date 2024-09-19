from wrap import wrap

def test__wrap():
    """The mutant does not append the last part of the text to lines, leading to an empty output for short texts."""
    output = wrap("Hello, World!", 20)
    assert len(output) == 1 and output[0] == "Hello, World!", "wrap must return the original string in a single-element list."