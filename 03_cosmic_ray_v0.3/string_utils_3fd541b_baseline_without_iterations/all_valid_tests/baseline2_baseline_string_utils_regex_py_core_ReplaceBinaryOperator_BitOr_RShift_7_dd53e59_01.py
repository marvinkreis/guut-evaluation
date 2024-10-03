from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Test string that should match the PRETTIFY_RE regex
    test_string = "This is a test string with a repeated space    and a question?  Is it valid?    Yes!"
    # The expected output after prettification should not contain multiple spaces or repeated question marks
    expected_output = "This is a test string with a repeated space and a question? Is it valid? Yes!"
    
    # Check if the pattern matches the expected string
    if PRETTIFY_RE['DUPLICATES'].search(test_string):
        test_string = PRETTIFY_RE['DUPLICATES'].sub(' ', test_string)  # Replace duplicates with a single space
    if PRETTIFY_RE['RIGHT_SPACE'].search(test_string):
        test_string = PRETTIFY_RE['RIGHT_SPACE'].sub(' ', test_string)  # Adjust spaces around signs

    # Assert that after processing, the output matches the expected output
    assert test_string == expected_output

test__prettify_re()