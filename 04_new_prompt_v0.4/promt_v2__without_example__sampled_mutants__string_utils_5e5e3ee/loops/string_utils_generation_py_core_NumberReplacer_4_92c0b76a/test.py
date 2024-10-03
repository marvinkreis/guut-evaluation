from string_utils.generation import roman_range

def test__mutant_killing_roman_range():
    """
    This test checks the output of the roman_range function when the stop value is set to 5.
    We expect the output to include 'I' when executing the baseline, as it starts from 1,
    whereas the mutant, which starts from 2, will not include 'I'.
    Thus, the test will pass for the baseline and fail for the mutant.
    """
    output = list(roman_range(5))
    assert output == ['I', 'II', 'III', 'IV', 'V'], f"Expected ['I', 'II', 'III', 'IV', 'V'], got {output}"