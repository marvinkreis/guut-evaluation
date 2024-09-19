from levenshtein import levenshtein

def test__levenshtein():
    """The mutant always returns a higher distance due to an incorrect addition when characters match."""
    output = levenshtein("abc", "abc")
    assert output == 0, "Levenshtein distance must be 0 for identical strings."