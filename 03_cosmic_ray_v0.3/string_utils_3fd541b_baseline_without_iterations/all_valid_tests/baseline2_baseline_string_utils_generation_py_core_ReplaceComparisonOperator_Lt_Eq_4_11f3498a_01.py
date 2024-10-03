from string_utils.generation import roman_range

def test_roman_range():
    # Testing a case that should raise an OverflowError with the original code
    try:
        # This should raise an OverflowError, as start (5) is not less than stop (5) with a negative step (-1)
        list(roman_range(stop=5, start=5, step=-1))
        assert False, "Expected OverflowError was not raised."
    except OverflowError:
        pass  # This is the expected outcome; the test should pass.

    # Also, test a valid case that should return proper roman numerals
    result = list(roman_range(stop=3, start=1, step=1))
    expected = ['I', 'II', 'III']
    assert result == expected, f"Expected {expected}, but got {result}."