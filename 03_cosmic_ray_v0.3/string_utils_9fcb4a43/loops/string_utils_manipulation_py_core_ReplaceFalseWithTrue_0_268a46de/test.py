from string_utils.manipulation import strip_html

def test__strip_html():
    """Changing the default value of 'keep_tag_content' in strip_html would cause it to incorrectly preserve HTML content."""
    input_string = 'test: <a href="foo/bar">click here</a>'
    
    correct_output = strip_html(input_string)
    mutant_output = strip_html(input_string, keep_tag_content=True)

    assert correct_output != mutant_output, "The outputs must differ between the correct and mutant implementations"