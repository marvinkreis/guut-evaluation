from string_utils.manipulation import slugify

def test_slugify():
    input_string = 'Hello   World  !'
    expected_output = 'hello-world'
    
    # Testing slugification
    output = slugify(input_string)
    
    # Assert that the output matches the expected output
    assert output == expected_output, f'Expected "{expected_output}" but got "{output}"'