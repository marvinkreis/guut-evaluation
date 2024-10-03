from string_utils.generation import roman_range

def test__roman_range_large_step():
    """
    Test the roman_range function with a large step size that exceeds the range. 
    The baseline should raise an OverflowError due to the step configuration,
    while the mutant should raise a ValueError while trying to yield an invalid Roman numeral.
    We assert that the baseline raises OverflowError and the mutant raises ValueError.
    """
    try:
        output = list(roman_range(start=1, stop=2, step=3))  # Step size is larger than the range
        # This line should never be reached
        assert False, "Expected OverflowError but did not receive it."
    except OverflowError:
        print("OverflowError correctly raised on baseline.")
    except ValueError:
        assert False, "ValueError should not be raised on baseline."