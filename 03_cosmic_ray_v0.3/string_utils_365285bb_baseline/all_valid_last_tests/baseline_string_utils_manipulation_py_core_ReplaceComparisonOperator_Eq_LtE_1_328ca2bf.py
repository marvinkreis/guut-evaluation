from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test for input 4 specifically, to expose the mutant
    output_4 = roman_encode(4)
    assert output_4 == 'IV', f"Expected 'IV' for input 4, but got '{output_4}'"  # This should fail on the mutant

    # Additional tests, focusing on a range that should work correctly
    assert roman_encode(1) == 'I', "Expected 'I' for input 1"
    assert roman_encode(2) == 'II', "Expected 'II' for input 2"
    assert roman_encode(3) == 'III', "Expected 'III' for input 3"
    assert roman_encode(5) == 'V', "Expected 'V' for input 5"
    assert roman_encode(6) == 'VI', "Expected 'VI' for input 6"
    assert roman_encode(7) == 'VII', "Expected 'VII' for input 7"
    assert roman_encode(8) == 'VIII', "Expected 'VIII' for input 8"
    assert roman_encode(9) == 'IX', "Expected 'IX' for input 9"
    assert roman_encode(10) == 'X', "Expected 'X' for input 10"

# Run the test function
test_roman_encode()