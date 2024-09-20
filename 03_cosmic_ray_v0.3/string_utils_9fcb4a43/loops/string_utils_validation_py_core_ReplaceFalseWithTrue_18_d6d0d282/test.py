from string_utils.validation import is_slug

def test__is_slug():
    """The mutant changed the behavior of is_slug, allowing invalid slugs such as an empty string."""
    assert not is_slug(''), "An empty string should not be a valid slug"
    assert not is_slug('my slug'), "A string with spaces should not be a valid slug"
    assert is_slug('my-blog-post-title'), "A correctly formatted slug should be valid"