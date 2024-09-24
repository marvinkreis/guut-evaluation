from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # Test case that includes multiple spaces and newlines
    test_string = "This is  a test with some whitespace:\n\n  This should   not get duplicated."

    # The expected output: newlines and multiple spaces should be reduced to a single space
    expected_result = "This is a test with some whitespace: This should not get duplicated."

    # Running the regex replace on the test string using the original behavior
    actual_result = PRETTIFY_RE['DUPLICATES'].sub(' ', test_string).strip()

    assert actual_result == expected_result, f"Expected: '{expected_result}', but got: '{actual_result}'"