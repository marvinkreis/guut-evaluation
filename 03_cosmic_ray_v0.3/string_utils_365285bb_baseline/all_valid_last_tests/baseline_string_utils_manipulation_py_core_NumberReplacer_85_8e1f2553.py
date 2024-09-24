from string_utils.manipulation import prettify

def test_prettify():
    # Example input with multiple formatting issues
    input_string = '   unprettified   string ,, like this one  ,will be"prettified"  .  It\'s awesome! '
    
    # Expected output after prettifying, correcting multiple spaces and punctuation
    expected_output = 'Unprettified string, like this one, will be "prettified". It\'s awesome!'
    
    # Run the prettify function
    output = prettify(input_string)

    # Assert that the output matches the expected output
    assert output == expected_output, f'Expected: {expected_output}, but got: {output}'

# Execute the test
test_prettify()