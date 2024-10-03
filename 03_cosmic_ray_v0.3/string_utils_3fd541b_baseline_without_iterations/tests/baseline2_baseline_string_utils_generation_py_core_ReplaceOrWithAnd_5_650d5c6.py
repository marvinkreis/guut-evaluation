from string_utils.generation import roman_range

def test_roman_range():
    # Test valid range
    output = list(roman_range(5))
    expected_output = ['I', 'II', 'III', 'IV', 'V']
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # Test invalid range with positive step
    try:
        list(roman_range(1, 5, 1))
    except OverflowError:
        pass  # Expected to raise OverflowError

    # Test invalid range with negative step
    try:
        list(roman_range(5, 1, -1))
    except OverflowError:
        pass  # Expected to raise OverflowError

    # Additional test for valid configuration
    output = list(roman_range(10, 1, 1))
    expected_output = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    # Test an edge case with zero step
    try:
        list(roman_range(10, 1, 0))
    except ValueError:
        pass  # Expected to raise ValueError due to zero step