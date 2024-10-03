from string_utils.generation import roman_range

def test_roman_range_mutant_killing():
    """
    Test the roman_range function using invalid backward configurations.
    The baseline will generate the sequence ['III', 'II'] for invalid parameters,
    while the mutant will raise an OverflowError for the same parameters.
    """
    # Expect the baseline to produce output and the mutant to raise an OverflowError
    output = list(roman_range(stop=2, start=3, step=-1))
    assert output == ['III', 'II'], f"Expected ['III', 'II'], but got: {output}"