from string_utils.generation import roman_range

def test__roman_range_invalid_with_step_one():
    """
    Test the roman_range function with a start greater than stop using a step of 1.
    The mutant should raise a ValueError due to invalid input, while the baseline should 
    raise an OverflowError for invalid range configuration.
    """
    try:
        # This configuration is expected to raise an OverflowError
        output = list(roman_range(1, 2, 1))
    except OverflowError as e:
        print(f"Raised OverflowError as expected: {str(e)}")
        return

    # If no error was raised, this is unexpected
    assert False  # Test should fail if no exception was raised