from string_utils.validation import is_slug

def test_is_slug():
    # Valid slugs (should return True)
    assert is_slug('my-blog-post-title') == True  # Valid slug
    assert is_slug('myblogtitle') == True          # No separators, valid
    assert is_slug('slug-1-valid') == True         # Numbers included
    
    # Invalid slugs (should return False)
    assert is_slug('Invalid Slug') == False        # Spaces should not be allowed
    assert is_slug('') == False                     # Empty string should be invalid
    assert is_slug('--invalid-title') == False     # Leading hyphens should be invalid
    assert is_slug('invalid-title--') == False     # Trailing hyphens should be invalid
    assert is_slug('myblog_post') == False         # Underscores should not be allowed
    assert is_slug('my blog post') == False        # Spaces should not be allowed

# Running the test
try:
    test_is_slug()
    print("All tests passed!")
except AssertionError:
    print("A test failed.")