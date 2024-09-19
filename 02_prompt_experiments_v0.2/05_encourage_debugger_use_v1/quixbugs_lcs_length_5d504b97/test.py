from lcs_length import lcs_length

def test__lcs_length():
    """The mutant changes the logic, causing it to produce incorrect lengths of common substrings."""
    
    # Test cases that are known to produce specific expected outputs
    assert lcs_length('witch', 'sandwich') == 2, "Expected length is 2"
    assert lcs_length('meow', 'homeowner') == 4, "Expected length is 4"
    assert lcs_length('abcde', 'abfce') == 2, "Expected length is 2"
    assert lcs_length('abcdef', 'zbcdefg') == 5, "Expected length is 5"