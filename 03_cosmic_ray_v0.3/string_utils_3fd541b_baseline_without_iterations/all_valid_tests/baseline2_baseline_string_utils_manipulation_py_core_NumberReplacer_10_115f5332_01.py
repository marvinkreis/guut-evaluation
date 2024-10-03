from string_utils.manipulation import roman_encode

def test__roman_encode_with_hundreds():
    # This input is specifically chosen because it falls into the hundreds range
    # With the original code, 300 should return 'CCC'
    # With the mutant code, it should return 'CC'
    correct_output = 'CCC'  # Expected output from the original implementation
    mutant_output = 'CC'     # Expected output from the mutant implementation

    # Test with input number 300
    result = roman_encode(300)
    
    # Check that the result is as expected for the original implementation
    assert result == correct_output, f"Expected {correct_output}, but got {result}"

# Note: When running with the mutant, the assertion would fail since it would return 'CC'.