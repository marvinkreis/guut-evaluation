from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test various numbers around the critical mutation point
    assert roman_encode(1) == 'I', "Expected output for roman_encode(1) to be 'I'"
    assert roman_encode(2) == 'II', "Expected output for roman_encode(2) to be 'II'"
    assert roman_encode(3) == 'III', "Expected output for roman_encode(3) to be 'III'"
    assert roman_encode(4) == 'IV', "Expected output for roman_encode(4) to be 'IV'"
    
    # Critical test for the number 5
    # This is the key point to see failure due to mutant
    expected_output_for_5 = 'V'
    assert roman_encode(5) == expected_output_for_5, "Expected output for roman_encode(5) to be 'V'"
    
    # Test number 6 which should return 'VI'
    assert roman_encode(6) == 'VI', "Expected output for roman_encode(6) to be 'VI'"

    # Checking a higher number
    assert roman_encode(10) == 'X', "Expected output for roman_encode(10) to be 'X'"

    # Edge case: Number 0 should raise ValueError
    try:
        roman_encode(0)
        assert False, "Expected ValueError for roman_encode(0), but none was raised."
    except ValueError:
        pass  # Expected behavior

    # Edge case: Number 4000 should raise ValueError
    try:
        roman_encode(4000)
        assert False, "Expected ValueError for roman_encode(4000), but none was raised."
    except ValueError:
        pass  # Expected behavior