from string_utils.manipulation import prettify

def test__prettify():
    """The mutant introduces an error in the __uppercase_first_letter_after_sign method, causing prettify to fail."""
    input_string = ' unprettified string ,, like this one,will be"prettified" .it\' s awesome! '
    output = prettify(input_string)
    
    # Since we know the correct output, we can check directly.
    expected_output = 'Unprettified string, like this one, will be "prettified". It\'s awesome!'
    assert output == expected_output, "prettify did not return the expected output"