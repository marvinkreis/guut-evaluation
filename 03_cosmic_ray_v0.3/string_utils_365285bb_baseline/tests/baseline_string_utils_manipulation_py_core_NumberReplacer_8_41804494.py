from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test case for encoding an integer that falls into the hundreds range
    # The expected output for 200 is 'CC', which should fail with the mutant due to incorrect mapping
    input_value = 200
    expected_output = 'CC'  # Correct encoding for 200 in Roman numerals
    actual_output = roman_encode(input_value)
    
    # Assert that the actual output matches the expected output
    assert actual_output == expected_output, f"Expected {expected_output}, but got {actual_output}"