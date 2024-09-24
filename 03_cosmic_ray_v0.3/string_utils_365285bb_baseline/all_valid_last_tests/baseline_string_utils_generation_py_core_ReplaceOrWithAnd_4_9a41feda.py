from string_utils.generation import roman_range

def test_roman_range_exceed():
    # Test with a valid backward configuration (should succeed)
    result = list(roman_range(start=5, stop=1, step=-1))
    assert result == ['V', 'IV', 'III', 'II', 'I'], f"Expected ['V', 'IV', 'III', 'II', 'I'], got {result}"

    # 1. Invalid upward configuration, expect OverflowError in original
    try:
        list(roman_range(start=1, stop=5, step=-1))  # Cannot go from 1 to 5 with a step of -1
        assert False, "Expected OverflowError was not raised."
    except OverflowError:
        pass  # This is expected

    # 2. Invalid backward configuration (step too small, valid path)
    try:
        # This should raise an OverflowError - going backward with step size not fitting
        list(roman_range(start=2, stop=1, step=-1))  # Should generate II -> I
        result = list(roman_range(start=2, stop=1, step=-1))
        assert result == ['II', 'I'], f"Expected ['II', 'I'], got {result}"
    except OverflowError:
        assert False, "Should not have raised OverflowError."

    # 3. Check a backward range that goes out of bounds
    try:
        list(roman_range(start=4, stop=1, step=-4))  # Cannot accommodate -4 stepping from 4 to 1
        assert False, "Expected OverflowError was not raised."
    except OverflowError:
        pass  # This is expected 

    # 4. Valid upward range
    result = list(roman_range(start=1, stop=4, step=1))
    assert result == ['I', 'II', 'III', 'IV'], f"Expected ['I', 'II', 'III', 'IV'], got {result}"

    # 5. Invalid backward configuration with a too large step
    try:
        list(roman_range(start=3, stop=1, step=-3))  # This should raise since step goes too large
        assert False, "Expected OverflowError was not raised."
    except OverflowError:
        pass  # This is expected

    # 6. Ensure proper handling with forward step that leads to invalid backward range
    try:
        list(roman_range(start=5, stop=1, step=1))  # This should lead to direct failure
        assert False, "Expected OverflowError was not raised."
    except OverflowError:
        pass  # This is expected