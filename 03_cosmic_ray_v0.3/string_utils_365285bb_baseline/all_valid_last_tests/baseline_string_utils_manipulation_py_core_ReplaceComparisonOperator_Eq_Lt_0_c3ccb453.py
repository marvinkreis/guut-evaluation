from string_utils.manipulation import roman_encode

def test_roman_encode():
    # Test case for input 0
    try:
        result = roman_encode(0)
        assert False, "Expected ValueError for input 0 but none was raised."
    except ValueError as e:
        assert str(e) == 'Input must be >= 1 and <= 3999', f"Expected 'Input must be >= 1 and <= 3999' for input 0, but got: {e}"

    # Test case for negative input -1
    try:
        result = roman_encode(-1)
        assert False, "Expected ValueError for input -1 but none was raised."
    except ValueError as e:
        assert str(e) == 'Input must be >= 1 and <= 3999', f"Expected 'Input must be >= 1 and <= 3999' for input -1, but got: {e}"

    # Valid test cases for Roman numeral conversions
    valid_cases = {
        1: 'I',
        2: 'II',
        3: 'III',
        4: 'IV',
        5: 'V',
        6: 'VI',
        7: 'VII',
        8: 'VIII',
        9: 'IX',
        10: 'X',
        50: 'L',
        100: 'C',
        500: 'D',
        1000: 'M',
        3999: 'MMMCMXCIX'
    }

    for number, expected_roman in valid_cases.items():
        result = roman_encode(number)
        assert result == expected_roman, f"Expected roman_encode({number}) to return '{expected_roman}', but got '{result}'"


# Run the test
if __name__ == "__main__":
    test_roman_encode()