from string_utils.validation import is_slug

def test__is_slug():
    """
    Test whether the function correctly identifies a slug string.
    The input is a valid slug 'my-blog-post-title' which should return true.
    This will fail if the mutant incorrectly alters the regex pattern for slug validation.
    """
    output = is_slug('my-blog-post-title')
    assert output == True