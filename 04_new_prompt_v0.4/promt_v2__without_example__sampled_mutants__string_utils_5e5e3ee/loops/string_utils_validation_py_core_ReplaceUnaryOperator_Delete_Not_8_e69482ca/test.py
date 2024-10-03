from string_utils.validation import is_pangram

def test__is_pangram_with_valid_and_partial_cases():
    """
    Test the is_pangram function with a full pangram and a partial string input.
    The full pangram "The quick brown fox jumps over the lazy dog" should return True,
    while the partial string "The quick brown fox" should return False in the baseline 
    and True in the mutant due to its incorrect logic.
    """
    # This input should return True in the baseline (valid pangram).
    assert is_pangram("The quick brown fox jumps over the lazy dog") == True 

    # This input should return False in the baseline (not a pangram).
    assert is_pangram("The quick brown fox") == False  # Baseline: False, Mutant: True