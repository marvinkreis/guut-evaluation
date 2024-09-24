from string_utils.generation import roman_range

def test__roman_range():
    # Normal case - should pass
    result = list(roman_range(7))
    expected = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    assert result == expected, f'Expected {expected}, but got {result}'

    # Case 1: Start > Stop (should raise OverflowError)
    try:
        list(roman_range(5, start=10, step=1))  # Invalid range (10 to 5)
        assert False, "Expected OverflowError (10 > 5) but did not raise"
    except OverflowError:
        pass  # Expected behavior

    # Case 2: Start equals Stop (should raise OverflowError)
    try:
        list(roman_range(3, start=3, step=1))  # Start == Stop should raise error
        assert False, "Expected OverflowError (3 == 3) but did not raise"
    except OverflowError:
        pass  # Expected behavior

    # Case 3: Valid backward range (should give expected result)
    result_backward = list(roman_range(1, start=5, step=-1))
    expected_backward = ['V', 'IV', 'III', 'II', 'I']
    assert result_backward == expected_backward, f'Expected {expected_backward}, but got {result_backward}'

    # Case 4: Invalid backward range (should raise OverflowError)
    try:
        list(roman_range(5, start=1, step=-1))  # Invalid case (1 cannot reach 5)
        assert False, "Expected OverflowError (1 to 5 backwards) but did not raise"
    except OverflowError:
        pass  # Expected behavior

    # Case 5: Check for an invalid zero step (should raise ValueError)
    try:
        list(roman_range(3, start=1, step=0))  # Invalid step
        assert False, "Expected ValueError for step of 0 but did not raise"
    except ValueError:
        pass  # Expected behavior

    # Extra case to see if the mutant allows an invalid forward case
    try:
        list(roman_range(3, start=4, step=1))  # Should raise because 4 cannot reach 3
        assert False, "Expected OverflowError (4 to 3) but did not raise"
    except OverflowError:
        pass  # Expected behavior