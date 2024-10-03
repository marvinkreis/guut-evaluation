from string_utils.manipulation import roman_encode

def test__roman_encode():
    # Test case for encoding 400, which should return 'CD'
    input_number = 400
    expected_output = 'CD'
    
    # Perform the encoding
    output = roman_encode(input_number)
    
    # Assert the output matches the expected output
    assert output == expected_output, f"Expected {expected_output} but got {output}"