from string_utils.generation import roman_range

def test__roman_range_behavior(mode='baseline'):
    """
    Test the roman_range for both baseline and mutant.
    :param mode: Specify 'baseline' or 'mutant' to run the relevant checks.
    """
    if mode == 'baseline':
        # Expected behavior for baseline
        output = list(roman_range(3999))
        assert output[-1] == 'MMMCMXCIX', f"Expected last numeral to be 'MMMCMXCIX', but got {output[-1]}"
        
    elif mode == 'mutant':
        # Expected behavior for mutant
        try:
            list(roman_range(3999))
            assert False, "Expected ValueError not raised for mutant."
        except ValueError as e:
            assert str(e) == '"stop" must be an integer in the range 1-3999', f"Unexpected error message: {str(e)}"
    else:
        raise ValueError("Invalid mode specified, must be 'baseline' or 'mutant'.")

# For executing tests, you could do something like:
# test__roman_range_behavior(mode='baseline')  # run this when the baseline is active
# test__roman_range_behavior(mode='mutant')    # run this when mutant is loaded