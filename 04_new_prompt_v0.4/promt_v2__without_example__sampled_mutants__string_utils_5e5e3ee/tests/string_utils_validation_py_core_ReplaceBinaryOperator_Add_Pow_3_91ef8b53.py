from string_utils.validation import is_slug

def test__is_slug_valid():
    """
    Test the is_slug function for a valid slug input. This test checks that the slug is valid in the baseline
    and should produce an error in the mutant due to a TypeError caused by the incorrect use of the `**` operator
    in the regex expression.
    """
    output = is_slug('my-blog-post-title')
    assert output == True  # Expecting True for a valid slug