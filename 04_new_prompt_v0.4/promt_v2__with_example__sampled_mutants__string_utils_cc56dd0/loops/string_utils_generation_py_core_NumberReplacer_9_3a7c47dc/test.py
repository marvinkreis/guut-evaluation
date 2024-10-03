from string_utils.generation import roman_range

def test_roman_range_mutant_killing():
    """
    Test the roman_range function with a stop value of 0. The baseline should raise a ValueError
    due to invalid input (must be in the range 1-3999), while the mutant will raise an OverflowError
    due to the changes in the validation logic.
    """
    try:
        # This should trigger the exception
        output = list(roman_range(stop=0))
        assert False, f"Expected a ValueError or OverflowError, but got an output: {output}"
    except ValueError as ve:
        # Expected behavior in the baseline
        print("Caught expected ValueError in baseline:", ve)
    except OverflowError as oe:
        # Expected behavior in the mutant
        print("Caught expected OverflowError in mutant:", oe)
        raise AssertionError("Mutant behavior detected: OverflowError")