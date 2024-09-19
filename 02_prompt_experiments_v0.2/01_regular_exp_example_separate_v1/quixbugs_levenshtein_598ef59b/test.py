from levenshtein import levenshtein

def test__levenshtein():
    """The mutant incorrectly computes the Levenshtein distance when characters differ.
    For the input ('kitten', 'sitting'), the correct implementation should return 3, 
    while the mutant will return 7 due to the erroneous addition when characters match."""
    correct_output = levenshtein("kitten", "sitting")
    assert correct_output == 3, "Levenshtein distance should be 3, but was computed incorrectly."