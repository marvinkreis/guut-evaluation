from string_utils.manipulation import prettify

def test_prettify_removes_internal_spaces():
    input_string = "This    is a   test  string."
    expected_output = "This is a test string."
    actual_output = prettify(input_string)
    assert actual_output == expected_output, f"Expected: '{expected_output}', but got: '{actual_output}'"