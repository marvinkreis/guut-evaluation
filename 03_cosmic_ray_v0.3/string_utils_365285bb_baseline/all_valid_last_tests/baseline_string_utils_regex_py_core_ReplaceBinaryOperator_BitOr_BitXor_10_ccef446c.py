from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # A complex input string with random spaces and newlines
    test_string = '  This   is a    test.    \n\n\nNew    line    follows.  \n\nLast line.   '

    # The expected output should remove excessive spaces within the text and newlines
    expected_output = 'This is a test. New line follows. Last line.'

    # Process the string using the PRETTIFY_RE regex
    processed_string = PRETTIFY_RE['DUPLICATES'].sub(' ', test_string).replace('\n', ' ').strip()

    # Assert that the processed string matches the expected output
    assert processed_string == expected_output, f"Expected: '{expected_output}', but got: '{processed_string}'"