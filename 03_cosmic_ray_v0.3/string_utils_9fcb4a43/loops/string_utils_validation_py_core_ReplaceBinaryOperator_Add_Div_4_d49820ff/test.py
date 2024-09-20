from string_utils.validation import is_slug

def test__is_slug():
    """The change in the regular expression in the mutant will cause it to raise an error, while the correct code returns True for a valid slug."""
    output = is_slug('my-blog-title')
    assert output is True, "is_slug should return True for valid slug"