from string_utils.generation import roman_range

def test__roman_range_backward_valid():
    """
    Test the roman_range function to ensure that it correctly handles valid backward ranges.
    Specifically, the case where `start` = 10, `stop` = 1, and `step` = -1 should generate
    a valid range of Roman numerals without raising errors in the baseline,
    while the mutant will throw an OverflowError due to its altered logic.
    """
    output = list(roman_range(start=10, stop=1, step=-1))
    expected_output = ['X', 'IX', 'VIII', 'VII', 'VI', 'V', 'IV', 'III', 'II', 'I']
    assert output == expected_output, f"Expected output {expected_output} but got {output}"