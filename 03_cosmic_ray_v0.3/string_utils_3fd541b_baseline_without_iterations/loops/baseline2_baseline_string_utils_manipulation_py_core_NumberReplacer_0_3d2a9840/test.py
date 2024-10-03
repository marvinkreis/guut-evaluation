from string_utils.manipulation import roman_encode

def test__roman_encode_mutant_detection():
    # Test case for the roman_encode function
    # This should return 'I' for the input 1, which is expected for valid roman numeral representation.
    correct_output = 'I'
    mutant_output = roman_encode(1)
    
    # Assert that the output from the correct implementation is equal to the expected output.
    assert mutant_output == correct_output, f"Expected {correct_output} but got {mutant_output} from the implementation"