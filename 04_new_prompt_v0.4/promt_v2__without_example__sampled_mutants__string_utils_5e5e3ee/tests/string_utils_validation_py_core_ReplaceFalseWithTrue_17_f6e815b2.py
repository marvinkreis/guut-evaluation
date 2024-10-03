from string_utils.validation import is_pangram

def test__is_pangram_mutant_killing():
    """
    Test whether the is_pangram function correctly identifies non-full strings.
    The input '' and ' ' should lead to different outputs between the baseline and the mutant,
    with the baseline returning False and the mutant returning True.
    """
    assert is_pangram('') == False
    assert is_pangram(' ') == False