from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test case for input of 0
    try:
        roman_encode(0)  # Should raise ValueError
        assert False, "Expected ValueError for input 0, but did not raise."
    except ValueError:
        pass  # This is expected for the original implementation

    # Test case for input of -1
    try:
        roman_encode(-1)  # Should raise ValueError
        assert False, "Expected ValueError for input -1, but did not raise."
    except ValueError:
        pass  # This is expected for the original implementation

    # Valid input tests
    assert roman_encode(1) == 'I', "Encoding for input 1 should be 'I'"
    assert roman_encode(2) == 'II', "Encoding for input 2 should be 'II'"
    assert roman_encode(3) == 'III', "Encoding for input 3 should be 'III'"
    assert roman_encode(4) == 'IV', "Encoding for input 4 should be 'IV'"
    assert roman_encode(5) == 'V', "Encoding for input 5 should be 'V'"
    
    # Test upper boundary input
    expected_output_3999 = 'MMMCMXCIX'
    actual_output_3999 = roman_encode(3999)
    assert actual_output_3999 == expected_output_3999, f"Expected '{expected_output_3999}' for input 3999, got '{actual_output_3999}'"

# Call the test function
test_roman_encode()