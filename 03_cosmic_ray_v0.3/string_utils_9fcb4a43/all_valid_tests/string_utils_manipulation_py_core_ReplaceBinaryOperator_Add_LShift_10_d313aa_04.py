from string_utils.manipulation import prettify

def test__prettify():
    """Test to specifically reveal the mutant's improper handling of string concatenation."""
    
    # Test case with heavy spacing and punctuation
    test_input_1 = "     This    is an  example  .   "
    expected_output_1 = "This is an example."
    correct_output_1 = prettify(test_input_1)
    
    # Validate output
    assert correct_output_1 == expected_output_1, "Expected fine formatting of spaces and punctuation."

    # More complex edge case
    test_input_2 = "    Hello!    How         are   you?   "
    expected_output_2 = "Hello! How are you?"
    correct_output_2 = prettify(test_input_2)
    
    # Validate output against expected
    assert correct_output_2 == expected_output_2, "Expected fine formatting of spaces and punctuation for edge case."

    # Edge case where the spacing might lead itself to generate an exception in the mutant
    test_input_edge = "   ....    Well   now   ? "
    expected_output_edge = ".... Well now?"
    correct_output_edge = prettify(test_input_edge)

    # Validate output for edge case
    assert correct_output_edge == expected_output_edge, "Expected proper spacing handling in edge cases."

# This test attempts to push handling of spacing around punctuation specifically to expose any issue with mutant logic.