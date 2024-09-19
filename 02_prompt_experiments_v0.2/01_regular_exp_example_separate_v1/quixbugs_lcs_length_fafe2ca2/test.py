from lcs_length import lcs_length

def test__lcs_length():
    """The mutant changes the logic to always compute fewer common substring lengths."""
    assert lcs_length('witch', 'sandwich') == 2, "Expected length of common substring is 2"
    assert lcs_length('meow', 'homeowner') == 4, "Expected length of common substring is 4"