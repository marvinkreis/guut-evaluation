from string_utils.generation import roman_range

def test__roman_range_valid_output():
    """
    Test roman_range with a valid stop value below 3999 to ensure proper output from baseline.
    """
    output = list(roman_range(10))  # Valid test that should run successfully
    assert output[-1] == 'X'  # Last value should be 'X'

def test__roman_range_invalid_upper_bound():
    """
    Test the roman_range function with a stop value of 4000.
    Both baseline and mutant should raise ValueError indicating the limit.
    """
    try:
        list(roman_range(4000))  # Should trigger ValueError in both versions
    except ValueError as e:
        # All implementations should handle this case
        assert str(e) == '"stop" must be an integer in the range 1-3999'
    else:
        raise AssertionError("Expected ValueError not raised on invalid upper bound input.")