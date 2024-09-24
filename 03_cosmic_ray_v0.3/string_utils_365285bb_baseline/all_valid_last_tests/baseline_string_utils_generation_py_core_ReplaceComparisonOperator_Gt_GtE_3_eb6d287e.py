from string_utils.generation import roman_range

def test_roman_range():
    # Test normal forward iteration
    result = list(roman_range(5))  # Should yield ['I', 'II', 'III', 'IV', 'V']
    expected = ['I', 'II', 'III', 'IV', 'V']
    assert result == expected, f"Expected {expected}, but got {result}"

    # Test backward iteration
    result = list(roman_range(stop=1, start=5, step=-1))  # Should yield ['V', 'IV', 'III', 'II', 'I']
    expected = ['V', 'IV', 'III', 'II', 'I']
    assert result == expected, f"Expected {expected}, but got {result}"

    # This test should raise an OverflowError with the original, but not with the mutant
    try:
        # In the original, this should raise an OverflowError because step would not allow valid iteration
        result = list(roman_range(stop=5, start=5, step=1))
        assert False, "Expected OverflowError but did not get one."
    except OverflowError:
        pass  # This is expected in original code

    # This specific case aims to test the mutant
    result = list(roman_range(stop=6, start=5, step=1))  # The original code should raise OverflowError
    expected = ['V', 'VI']  # The mutant will allow these values due to changed logic
    assert result == expected, f"Expected {expected}, but got {result} for start=5, stop=6, step=1"

    # Another test to ensure OverflowError is raised correctly in the original code
    try:
        _ = list(roman_range(1, start=2, step=1))  # Should raise OverflowError
        assert False, "Expected OverflowError but did not get one."
    except OverflowError:
        pass  # This is expected
        
    # Verify OverflowError on step too large
    try:
        _ = list(roman_range(5, start=6, step=1))  # Should raise OverflowError in the original code
        assert False, "Expected OverflowError but did not get one."
    except OverflowError:
        pass  # expected behavior