from string_utils.generation import roman_range

def test__roman_range():
    """
    Test whether the roman_range function correctly generates Roman numerals when the step is too large.
    Using start=1, stop=10, and step=20 should produce an OverflowError in the correct implementation
    as the conditions should not permit iteration. The mutant's condition replacement might fail to trigger 
    this error, thereby exposing the mutant.
    """
    try:
        output = list(roman_range(10, 1, 20))
        assert False, "Expected OverflowError but got a result: {}".format(output)
    except OverflowError:
        pass  # This indicates that the baseline implementation is correct with proper error handling