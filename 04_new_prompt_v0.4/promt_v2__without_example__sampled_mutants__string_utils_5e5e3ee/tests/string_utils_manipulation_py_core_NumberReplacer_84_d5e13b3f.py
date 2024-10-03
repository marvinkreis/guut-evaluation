from string_utils.manipulation import prettify

def test__string_formatter_duplicate_handling():
    """
    Test whether the StringFormatter handles multiple occurrences of placeholders correctly.
    The input contains a duplicate placeholder, and the mutant modifies the replacement behavior.
    The baseline should return the same string while the mutant should return a string with only the first occurrence replaced.
    """
    input_string = 'Check this first email john.doe@example.com and this second email john.doe@example.com.'
    expected_output_baseline = 'Check this first email john.doe@example.com and this second email john.doe@example.com.'
    output = prettify(input_string)
    assert output == expected_output_baseline, f"Expected: {expected_output_baseline}, but got: {output}"