from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test case for the mutant version where the condition for 0 is changed to check for negative values
    # The original encoding function should correctly return Roman numeral for 0
    # Since Roman numeral does not have a representation for 0, we expect a ValueError for input of 0
    try:
        roman_encode(0)
        assert False, "Expected ValueError for input 0 but did not get it."
    except ValueError:
        pass  # This is expected behavior for the original function

    # Checking with valid input
    result = roman_encode(1)
    assert result == 'I', f"Expected 'I', but got {result}"

    # Additional valid test cases
    result = roman_encode(4)
    assert result == 'IV', f"Expected 'IV', but got {result}"

    result = roman_encode(10)
    assert result == 'X', f"Expected 'X', but got {result}"

    result = roman_encode(2023)
    assert result == 'MMXXIII', f"Expected 'MMXXIII', but got {result}"