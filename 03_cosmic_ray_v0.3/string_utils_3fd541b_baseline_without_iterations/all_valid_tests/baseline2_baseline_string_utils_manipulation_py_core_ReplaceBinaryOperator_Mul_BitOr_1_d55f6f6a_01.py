from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test inputs known to have specific Roman numeral outputs
    input_value = 8
    correct_output = 'VIII'  # The expected Roman numeral for 8
    mutant_output = roman_encode(input_value)  # Runs through both the original and mutant
    
    # Assert the output is as expected for the original function
    assert mutant_output == correct_output, f"Expected {correct_output}, but got {mutant_output}"

    # Additionally, testing an edge case
    input_value = 9
    correct_output = 'IX'  # The expected Roman numeral for 9
    mutant_output = roman_encode(input_value)  # Runs through both the original and mutant
    
    # Assert the output is as expected for the original function
    assert mutant_output == correct_output, f"Expected {correct_output}, but got {mutant_output}"