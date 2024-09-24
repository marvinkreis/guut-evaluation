from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Testing the encoding of the number 9
    input_number = 9
    expected_output = 'IX'
    
    # Call the roman_encode function
    actual_output = roman_encode(input_number)
    
    # Assert that the actual output matches the expected output
    assert actual_output == expected_output, f"Expected {expected_output}, but got {actual_output}"