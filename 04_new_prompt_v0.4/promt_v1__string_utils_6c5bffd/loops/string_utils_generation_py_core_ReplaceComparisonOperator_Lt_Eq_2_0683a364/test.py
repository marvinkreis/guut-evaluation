from string_utils.generation import roman_range

def test__roman_range_start():
    """
    Test the roman_range function with start value set to 1 and stop value set to 5.
    The baseline should return ['I', 'II', 'III', 'IV', 'V'], while the mutant is expected 
    to raise a ValueError because the start cannot be equal to 1 according to the mutant logic.
    """
    output = list(roman_range(start=1, stop=5))  # For baseline, this should be valid
    assert output == ['I', 'II', 'III', 'IV', 'V']