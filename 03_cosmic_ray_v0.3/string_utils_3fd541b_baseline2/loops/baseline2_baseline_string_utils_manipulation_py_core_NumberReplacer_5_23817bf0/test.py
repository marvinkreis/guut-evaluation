from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Test case for the roman_encode function
    correct_output = 'X'  # The expected output for 10
    mutant_output = roman_encode(10)  # Get the output from the mutant
    
    # Correct implementation should return 'X' for 10
    assert mutant_output == correct_output, f"Expected '{correct_output}', but got '{mutant_output}'"