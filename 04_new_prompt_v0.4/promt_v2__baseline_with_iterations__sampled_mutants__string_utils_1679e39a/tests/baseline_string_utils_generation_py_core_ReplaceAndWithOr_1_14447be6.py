from string_utils.generation import roman_range

def test__roman_range():
    """
    Test the behavior of the roman_range function when stop is less than start with a negative step.
    The input (stop=1, start=7, step=-1) is expected to generate the roman numbers from VII to I.
    The mutant changes the backward_exceed condition such that it does not correctly identify the overflow case,
    thus allowing an invalid configuration to be processed which results in an incorrect output.
    """
    output = list(roman_range(stop=1, start=7, step=-1))
    assert output == ['VII', 'VI', 'V', 'IV', 'III', 'II', 'I']