from string_utils.generation import roman_range

def test_roman_range_mutant_killing():
    """
    Test the roman_range function with a maximum valid input configuration.
    The baseline should raise an OverflowError,
    while the mutant yields valid output for an invalid range configuration.
    """
    try:
        result = list(roman_range(3999, start=3999, step=-1))
        assert False, "Expected an OverflowError, but did not get one."
    except OverflowError:
        print("Correctly raised OverflowError on baseline.")
    except Exception as e:
        assert False, f"Unexpected exception raised: {str(e)}"