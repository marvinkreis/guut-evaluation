from string_utils.manipulation import roman_encode  # Adjusting the import to match the directory structure

def test_roman_encode():
    # Testing known values for correct code
    assert roman_encode(1) == 'I'   # Basic test
    assert roman_encode(4) == 'IV'  # Edge case, should fail on mutant
    assert roman_encode(5) == 'V'   # Should be correct
    assert roman_encode(6) == 'VI'  # Should be correct
    assert roman_encode(7) == 'VII' # This is our key test that should fail for the mutant
    assert roman_encode(8) == 'VIII'  # Should also be correct
    assert roman_encode(9) == 'IX'  # Another key test for mutation
    assert roman_encode(10) == 'X'  # Should be correct

    # Additional assertions to ensure proper functionality
    assert roman_encode(37) == 'XXXVII'  # Expanding test cases
    assert roman_encode(2021) == 'MMXXI'  # Test larger value

# Running the test
if __name__ == "__main__":
    test_roman_encode()
    print("All tests passed successfully.")