from lcs_length import lcs_length

def test__lcs_length():
    """Changing the indexing in lcs_length would result in incorrect lengths of common substrings."""
    assert lcs_length('abcdefgh', 'xyzabcde') == 5, "Should be 5 for common substring 'abcde'"
    assert lcs_length('abcdefgh', 'abcdefgh') == 8, "Should be 8 for the whole string match"