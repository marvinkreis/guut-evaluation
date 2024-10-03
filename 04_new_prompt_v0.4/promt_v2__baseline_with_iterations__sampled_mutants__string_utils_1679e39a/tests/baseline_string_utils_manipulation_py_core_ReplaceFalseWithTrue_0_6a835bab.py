from string_utils.manipulation import strip_html

def test__strip_html():
    """
    Test whether HTML tags are removed correctly from a string.
    The input contains an HTML anchor tag that should be entirely removed 
    when keep_tag_content is set to False. The baseline implementation 
    should correctly remove the content, while the mutant implementation 
    will generate a different output because it sets keep_tag_content to True by default.
    Therefore, we will check that the outputs are as expected for both implementations.
    """
    # Test case for the baseline functionality
    output_without_content = strip_html('test: <a href="foo/bar">click here</a>', keep_tag_content=False)
    assert output_without_content == 'test: '  # This should pass in baseline

    # Test the mutant explicitly with keep_tag_content set to True
    # This should output: 'test: click here'
    output_with_content = strip_html('test: <a href="foo/bar">click here</a>', keep_tag_content=True)
    assert output_with_content == 'test: click here'  # This should pass only in the mutant

    # Additional test to verify the default behavior (with no explicit keep_tag_content)
    # Since the mutant changes the default behavior to keep the tag content,
    #  this should again yield 'test: click here' on the mutant.
    output_default_behavior = strip_html('test: <a href="foo/bar">click here</a>')
    assert output_default_behavior == 'test: '  # This should pass on the baseline