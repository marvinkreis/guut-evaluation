from string_utils.manipulation import prettify

def test__saxon_genitive_detection():
    # This will have a saxon genitive case that is expected to be fixed by removing spaces between name and 's
    input_string = "Dave' s dog"
    expected_output = "Dave's dog"  # Correct expected output without space
    actual_output = prettify(input_string)
    
    # Asserting expected vs actual output
    assert actual_output == expected_output, f"Expected: '{expected_output}', but got: '{actual_output}'"

    # Check the mutant by providing input that is expected to fail distinctively with the mutant's code
    mutant_input = "John 's book"
    mutant_expected_output = "John's book"  # Expected output without space
    mutant_actual_output = prettify(mutant_input)

    # If the mutant is active, it would incorrectly add a space between John and the apostrophe.
    assert mutant_actual_output == mutant_expected_output, f"Mutant failed - Expected: '{mutant_expected_output}', but got: '{mutant_actual_output}'"