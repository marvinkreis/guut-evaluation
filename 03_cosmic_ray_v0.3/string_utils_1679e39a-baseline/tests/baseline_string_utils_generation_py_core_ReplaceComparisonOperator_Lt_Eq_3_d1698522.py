from string_utils.generation import roman_range

def test__roman_range():
    """
    Test the 'roman_range' function with a range that would normally generate 
    Roman numerals backwards but with a non-zero step. This condition should 
    raise an OverflowError in the baseline condition, but the mutant changes 
    the step condition check, allowing it to produce the range without 
    raising the error.
    """
    try:
        # This should raise an OverflowError because starting from 1 to 7 with a step of -1 
        # does not make sense when you want to go "backwards", as start is less than stop.
        list(roman_range(7, start=1, step=-1))
        assert False, "Expected an OverflowError but did not get one."
    except OverflowError:
        pass  # This is the expected behavior in the baseline code