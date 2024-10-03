from string_utils.validation import is_slug

def test__kill_mutant():
    """
    This test is designed to check if the string 'slug-with-numbers-123' is recognized as a valid slug.
    The baseline should return True, indicating it's a valid slug, while the mutant should return False due to the change 
    introduced in the is_slug function, thus killing the mutant.
    """
    input_slug = 'slug-with-numbers-123'
    output = is_slug(input_slug)
    print(f"Output for input '{input_slug}': {output}")
    assert output is True, "The output should have been True for a valid slug."