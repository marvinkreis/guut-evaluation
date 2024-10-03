from string_utils.generation import roman_range

def test_roman_range():
    # Test the valid forward range with step 1
    result = list(roman_range(5))
    expected = ['I', 'II', 'III', 'IV', 'V']
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test for a scenario where the mutant would fail
    # This call is supposed to work with the original code
    # but will fail with the mutant because it incorrectly
    # allows an invalid range setup where step goes from start to stop + step
    result = list(roman_range(10, start=5, step=5))
    expected = ['V', 'X']  # Only valid output given the original check
    assert result == expected, f"Expected {expected}, but got {result}"

# To execute the test manually, uncomment the following line
# test_roman_range()