from string_utils.generation import roman_range

def test__roman_range():
    """Mutant changes loop condition causing it to yield only the first numeral."""
    output = list(roman_range(5))
    assert len(output) > 1, "roman_range should produce multiple values."