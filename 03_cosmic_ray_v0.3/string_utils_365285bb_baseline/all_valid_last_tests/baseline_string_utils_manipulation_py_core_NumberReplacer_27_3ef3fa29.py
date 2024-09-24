from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Testing valid Roman numeral conversions
    assert roman_encode(1) == 'I', "Expected output for 1 is 'I'"   # 1 -> 'I'
    assert roman_encode(2) == 'II', "Expected output for 2 is 'II'" # 2 -> 'II'
    assert roman_encode(3) == 'III', "Expected output for 3 is 'III'" # 3 -> 'III'
    
    # Testing the critical values
    assert roman_encode(4) == 'IV', "Expected output for 4 is 'IV'"   # 4 -> 'IV'
    
    # This is the number directly tied to the mutant's alteration
    expected_for_five = 'V'  # This is what we expect for roman_encode(5)
    assert roman_encode(5) == expected_for_five, f"Expected output for 5 is {expected_for_five}"  # 5 should yield 'V'

    # Checking adjacent numeral to see correct logic handling
    assert roman_encode(6) == 'VI', "Expected output for 6 is 'VI'"  # 6 -> 'VI'
    
    # Pushing for testing blocks above the problematic areas
    assert roman_encode(9) == 'IX', "Expected output for 9 is 'IX'"    # 9 -> 'IX'
    assert roman_encode(10) == 'X', "Expected output for 10 is 'X'"    # 10 -> 'X'
    assert roman_encode(37) == 'XXXVII', "Expected output for 37 is 'XXXVII'"  # 37 -> 'XXXVII'

    # Ensure output is confirmed and not obscured by invalid checks
    print("All tests passed!")

# Execute the test
test_roman_encode()