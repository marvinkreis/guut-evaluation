from string_utils.generation import roman_range

def test_roman_range_mutant_detection():
    # 1. This should raise OverflowError for start > stop with a positive step
    try:
        list(roman_range(stop=1, start=5, step=1))  # Expecting OverflowError
        assert False, "Expected OverflowError not raised for start=5, stop=1, step=1"
    except OverflowError:
        pass  # This is expected

    # 2. This should produce a valid output: ['I', 'II', 'III', 'IV', 'V']
    valid_output = list(roman_range(stop=5, start=1, step=1))
    assert valid_output == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {valid_output}"

    # 3. Case with step=0: should raise ValueError
    try:
        list(roman_range(stop=5, start=3, step=0))  # Expects ValueError
        assert False, "Expected ValueError for step=0 not raised"
    except ValueError:
        pass  # This is what we expect

    # 4. Negative step from lower to higher should trigger OverflowError
    try:
        list(roman_range(stop=5, start=3, step=-1))  # Expects OverflowError
        assert False, "Expected OverflowError not raised for start=3, stop=5, step=-1"
    except OverflowError:
        pass  # This is expected

    # 5. Test a valid backward range
    valid_output = list(roman_range(stop=1, start=3, step=-1))  # Expects ['III', 'II', 'I']
    assert valid_output == ['III', 'II', 'I'], f"Expected ['III', 'II', 'I'], got {valid_output}"

    # 6. Check error when starting and stopping at the same value with a negative step
    try:
        list(roman_range(stop=1, start=1, step=-1))  # Should raise OverflowError
        assert False, "Expected OverflowError not raised for start=1, stop=1, step=-1"
    except OverflowError:
        pass  # Expected behavior

    # 7. Check another zero step error case
    try:
        list(roman_range(stop=1, start=2, step=0))  # Should raise ValueError
        assert False, "Expected ValueError for zero step not raised for start=2, stop=1"
    except ValueError:
        pass  # This should also be expected
