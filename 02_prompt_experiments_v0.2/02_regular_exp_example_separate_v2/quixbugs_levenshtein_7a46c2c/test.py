from levenshtein import levenshtein

def test__levenshtein():
    """Changing the return value when characters match would invalidate the distance calculation."""
    output = levenshtein("example", "example")
    assert output == 0, "the distance should be 0 for identical strings"
    
    output_different = levenshtein("example", "samples")
    assert output_different == 3, "the distance should be 3 for 'example' and 'samples'"