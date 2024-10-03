from string_utils.manipulation import prettify

def test__prettify_with_space_around():
    # Test the prettify function which indirectly calls __ensure_spaces_around
    input_string = 'This is  a test.   This should be cleaned up.   '
    expected_output = 'This is a test. This should be cleaned up.'
    assert prettify(input_string) == expected_output, f"Expected: '{expected_output}', but got: '{prettify(input_string)}'"