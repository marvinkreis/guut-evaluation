from string_utils.generation import roman_range

def test__roman_range():
    """
    Test for the proper functioning of the roman_range generator for valid parameters, specifically when the start is less than stop,
    with a step of 1. The input (start=1, stop=5, step=1) should correctly yield the roman numerals from 'I' to 'V'.
    The mutant modifies the condition for determining excess in forward generation, which will cause it to mishandle cases where
    the start is exactly less than stop.
    """
    output = list(roman_range(5, 1, 1))  # should yield ['I', 'II', 'III', 'IV', 'V']
    assert output == ['I', 'II', 'III', 'IV', 'V']