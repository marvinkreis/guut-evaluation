from string_utils.manipulation import prettify

def test__prettify():
    # Test string that requires prettification and has spaces that should be handled
    input_string = " unprettified string , like this one , will be prettified . it' s awesome! "
    
    # Expected output after correct prettification
    expected_output = 'Unprettified string, like this one, will be prettified. It\'s awesome!'

    # Running the prettify function
    actual_output = prettify(input_string)

    # Assert that the actual output matches the expected output
    assert actual_output == expected_output, f"Expected: {expected_output}, but got: {actual_output}"