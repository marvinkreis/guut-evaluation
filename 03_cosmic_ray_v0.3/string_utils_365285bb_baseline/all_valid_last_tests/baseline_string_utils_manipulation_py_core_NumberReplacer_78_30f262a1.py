from string_utils.manipulation import prettify

def test_prettify():
    # Test Case 1: Regular input with additional spaces
    input_string = "   This   is   a    test.   Check   for   spaces!   "
    expected_output = "This is a test. Check for spaces!"
    
    # Assert the output from the original code
    assert prettify(input_string) == expected_output

    # Test Case 2: Intentionally flawed input
    mutant_input = "     This   input has multiple spaces     .   "
    
    # Expected output after prettification
    original_expected_output = "This input has multiple spaces."
    
    # Assert that the original implementation handles the input correctly
    assert prettify(mutant_input) == original_expected_output

    # Define the mutant's expected output based on the faulty behavior
    mutant_expected_output = " This   input has multiple spaces .  "  # Example of handling from the mutant

    # Assert that the mutant does NOT produce the same (correct) output
    assert prettify(mutant_input) != mutant_expected_output
