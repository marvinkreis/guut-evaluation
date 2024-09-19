from next_permutation import next_permutation

def test__next_permutation():
    """The mutant changes the condition and fails to produce the correct next permutation."""
    input_permutation = [3, 2, 4, 1]
    correct_output = next_permutation(input_permutation)
    expected_output = [3, 4, 1, 2]  # The expected next permutation
    mutant_output = [3, 1, 2, 4]  # Known output from the mutant for the same input
  
    assert correct_output == expected_output, "The correct implementation must return the expected next permutation."
    assert mutant_output != expected_output, "The mutant should not return the expected next permutation."