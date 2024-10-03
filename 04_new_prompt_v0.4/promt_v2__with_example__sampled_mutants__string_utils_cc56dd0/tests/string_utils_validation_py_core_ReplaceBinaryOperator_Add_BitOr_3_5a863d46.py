from string_utils.validation import is_slug

def test_is_slug_mutant_killing():
    """
    Test the is_slug function with a valid slug. The mutant will raise a TypeError
    due to the incorrect regex construction, while the baseline will return True for a valid slug.
    """
    output = is_slug('my-blog-post-title')
    assert output == True, f"Expected True, got {output}"