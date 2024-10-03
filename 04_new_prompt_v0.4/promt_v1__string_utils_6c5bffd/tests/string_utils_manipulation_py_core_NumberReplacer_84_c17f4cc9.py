from string_utils.manipulation import prettify

def test__prettify_multiple_placeholders_kill_mutant():
    """
    This test checks the behavior of the prettify function when the input contains multiple identical URLs.
    The mutant should fail because it improperly replaces the second occurrence of the URL with a placeholder.
    The expected output should contain both URLs unchanged.
    """
    input_string = 'Check this out: https://example.com and also visit https://example.com.'
    output = prettify(input_string)
    assert output == 'Check this out: https://example.com and also visit https://example.com.'