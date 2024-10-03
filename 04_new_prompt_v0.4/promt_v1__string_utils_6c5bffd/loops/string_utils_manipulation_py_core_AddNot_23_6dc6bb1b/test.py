from string_utils.manipulation import strip_html

def test__strip_html():
    """
    Test the behavior of the strip_html function with keep_tag_content parameter.
    The mutant alters the expected behavior such that when keep_tag_content is True, 
    it removes the HTML tag content instead of keeping it.
    """
    input_string = 'test: <a href="foo/bar">click here</a>'
    expected_with_content = 'test: click here'
    expected_without_content = 'test: '

    # Test with keep_tag_content=True
    output_with_content = strip_html(input_string, keep_tag_content=True)
    assert output_with_content == expected_with_content

    # Test with keep_tag_content=False
    output_without_content = strip_html(input_string, keep_tag_content=False)
    assert output_without_content == expected_without_content