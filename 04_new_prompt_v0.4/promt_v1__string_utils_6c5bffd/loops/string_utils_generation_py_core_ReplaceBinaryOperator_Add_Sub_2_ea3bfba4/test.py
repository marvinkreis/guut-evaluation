from string_utils.generation import roman_range

def test__roman_range_invalid_step_configuration():
    """
    Test whether the roman_range function correctly raises an OverflowError for invalid step configuration.
    The input (start=1, stop=1, step=-1) is expected to raise an OverflowError in the Baseline, 
    but not in the Mutant version due to an altered validation logic.
    """
    try:
        output = list(roman_range(1, 1, -1))  # This should raise an OverflowError
        assert False, "Expected OverflowError was not raised."
    except OverflowError as e:
        print(f"OverflowError caught: {e}")  # Should catch and print the expected error