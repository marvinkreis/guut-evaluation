from string_utils.manipulation import prettify

def test__prettify_with_url():
    """
    Test whether the prettify function correctly restores URLs. The input includes a URL,
    which should be restored in the output for the baseline, but not for the mutant,
    resulting in different outputs that will kill the mutant.
    """
    input_string = 'Visit us at https://example.com for more information.'
    output = prettify(input_string)
    assert output == 'Visit us at https://example.com for more information.'  # Expected behavior for the baseline.