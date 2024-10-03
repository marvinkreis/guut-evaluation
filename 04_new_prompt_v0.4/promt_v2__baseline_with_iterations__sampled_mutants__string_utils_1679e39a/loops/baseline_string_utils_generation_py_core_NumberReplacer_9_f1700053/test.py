from string_utils.generation import roman_range

def test__roman_range():
    """
    Test whether the validation of argument values in the roman_range function correctly raises an error 
    when a negative stop value is provided. The correct implementation should raise a ValueError 
    for stop < 1, while the mutant incorrectly allows stop to be 0 or negative. This test will fail 
    when running against the mutant and pass against the original implementation.
    """
    try:
        list(roman_range(0))  # Attempt to create a roman range with stop value of 0
    except ValueError as e:
        assert str(e) == '"stop" must be an integer in the range 1-3999'
    else:
        assert False, "Expected a ValueError for stop value of 0, but no exception was raised."