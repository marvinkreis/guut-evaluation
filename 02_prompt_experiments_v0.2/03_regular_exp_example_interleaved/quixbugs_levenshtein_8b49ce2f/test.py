from levenshtein import levenshtein

def test__levenshtein():
    """Mutant changes output when first characters are equal. This test checks the calculation."""
    output = levenshtein("abc", "abc")
    assert output == 0, "levenshtein should return 0 for equal strings"