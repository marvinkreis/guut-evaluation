from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """Test to detect differences in ISBN validation between correct code and mutant."""
    
    # Valid ISBN
    assert is_isbn_13('978-0-306-40615-7'), "Expected True for valid ISBN."
    
    # Invalid ISBN with an incorrect checksum digit
    assert not is_isbn_13('978-0-306-40615-8'), "Expected False for invalid ISBN due to checksum."

    # Malformed ISBN with a non-numeric character
    assert not is_isbn_13('978-0-306-4A615-7'), "Expected False for invalid ISBN with non-numeric characters."