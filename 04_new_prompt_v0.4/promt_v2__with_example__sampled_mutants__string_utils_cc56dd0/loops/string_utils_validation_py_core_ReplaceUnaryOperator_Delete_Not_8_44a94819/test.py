from string_utils.validation import is_pangram

def test_is_pangram_mutant_killing():
    """
    Test the is_pangram function with a known pangram. The baseline should return True since the input
    is a valid pangram, while the mutant will incorrectly return False.
    """
    output = is_pangram("The quick brown fox jumps over the lazy dog")
    assert output == True, f"Expected True, got {output}"