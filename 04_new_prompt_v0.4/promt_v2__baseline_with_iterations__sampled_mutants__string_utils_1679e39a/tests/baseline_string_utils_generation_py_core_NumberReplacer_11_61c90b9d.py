from string_utils.generation import roman_range

def test__roman_range():
    """
    Test the upper bound of the roman_range function. The input is set to 3999, which is the maximum acceptable
    value according to the original code. The mutant decreases the upper bound to 3998, meaning if we attempt to 
    generate roman numerals up to 3999, the mutant will not raise an error but the output will stop at 3998 instead.
    This test will pass with the correct implementation but will fail with the mutant.
    """
    output = list(roman_range(3999))
    expected_output = [...]  # Define the expected output for numbers 1 to 3999 in Roman numerals
    # Due to the limitation, we can only test for a few values and ensure that the list at least contains the last value.
    assert output[-1] == 'MMMCMXCIX'  # 3999 in Roman numerals