from string_utils.generation import roman_range

def test__roman_range():
    """
    Test the default behavior of roman_range when called without a start parameter.
    The mutant starts generating Roman numerals from 'II', while the baseline starts from 'I'.
    This test checks for the difference in output based on the changed default start value.
    """
    output = list(roman_range(7))
    assert output == ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], f"Expected ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'], got {output}"