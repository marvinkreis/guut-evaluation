from string_utils.validation import is_slug

def test__is_slug():
    # Test with an invalid slug (empty string)
    result = is_slug('')
    assert result == False, f'Expected False but got {result}'

    # Test with a space string
    result = is_slug(' ')
    assert result == False, f'Expected False but got {result}'

    # Test with non-snake case string
    result = is_slug('Invalid-Slug-!')
    assert result == False, f'Expected False but got {result}'

    # Test with valid slug
    result = is_slug('valid-slug-title')
    assert result == True, f'Expected True but got {result}'

    # Test with a valid slug that has numbers
    result = is_slug('slug123')
    assert result == True, f'Expected True but got {result}'