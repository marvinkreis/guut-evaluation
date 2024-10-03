from string_utils.validation import is_pangram

def test__is_pangram_mutant_killer():
    """
    Test the is_pangram function specifically to expose the mutant's incorrect logic.
    Using valid pangram inputs to demonstrate how the mutant behaves incorrectly 
    compared to the baseline.
    """
    # A valid pangram that should return True in the baseline
    valid_pangram = "The quick brown fox jumps over the lazy dog"
    assert is_pangram(valid_pangram) == True  # This should pass

    # Another valid pangram that should also return True
    another_valid_pangram = "A wizard's job is to vex chumps quickly in fog"
    assert is_pangram(another_valid_pangram) == True  # This should also pass

    # Edge case input (spaces only)
    edge_case_input = "   "
    output_edge_case = is_pangram(edge_case_input)
    print(f'Edge case input (whitespace): Output: {output_edge_case}, Expected: False')
    assert output_edge_case == False  # should pass in both, and will expose mutant behavior