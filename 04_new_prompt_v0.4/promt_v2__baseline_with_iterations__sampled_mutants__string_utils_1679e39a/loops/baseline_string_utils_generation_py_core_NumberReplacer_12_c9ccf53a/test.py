from string_utils.generation import roman_range

def test__roman_range():
    """
    Test the roman_range function to ensure that it correctly generates Roman numerals
    from 1 to 5 with a step of 1. This input (start=1, stop=5, step=1) is valid 
    in both implementations, but only the original code will handle the range 
    correctly. The mutant's condition will allow for an invalid configuration 
    due to the modified forward_exceed logic.
    """
    gen = roman_range(stop=5, start=1, step=1)
    output = list(gen)
    assert output == ['I', 'II', 'III', 'IV', 'V']  # Correct Roman numerals up to 5

    # Now testing an invalid configuration where the mutant will fail
    try:
        gen_invalid = roman_range(stop=1, start=10, step=1)  # Start > Stop should raise OverflowError
        list(gen_invalid)  # Should not reach here
        assert False, "The mutant should raise an OverflowError for this invalid configuration."
    except OverflowError:
        # This confirms that the mutant's logic does not handle this case properly
        pass