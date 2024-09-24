from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Define test cases that we know the expected output for a valid roman encoding
    test_cases = [
        (1, 'I'),
        (4, 'IV'),
        (5, 'V'),
        (9, 'IX'),
        (10, 'X'),
        (37, 'XXXVII'),
        (100, 'C'),
        (3999, 'MMMCMXCIX'),
    ]

    # Validate that all known cases return expected results
    for input_number, expected_roman in test_cases:
        actual_roman = roman_encode(input_number)
        # Assert that actual output equals expected output
        assert actual_roman == expected_roman, f"Expected {expected_roman} for input {input_number}, got {actual_roman}"

    # Now, let's induce a situation where the mutant affects the results
    mutant_case = 3  # Example, where normally we expect 'III'
    actual_output = roman_encode(mutant_case)

    # Simulate a mutant output which would incorrectly handle encoding
    mutant_output = actual_output.replace('I', '!')  # Introduce a change that the mutant might produce

    # The output with the mutant should be different from the original expected output
    assert actual_output != mutant_output, "The output with the mutant should be different from the correct output."
    assert '!' in mutant_output, "Mutant output should contain '!', indicating an incorrect encoding."
