from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """The mutant incorrectly allows an empty string as a valid ISBN-13; this test should fail if the mutant is present."""
    valid_isbn = '9780312498580'  # Valid ISBN-13
    invalid_isbn = '9780312498581'  # Invalid ISBN-13
    empty_isbn = ''  # Empty string

    # Valid input test
    assert is_isbn_13(valid_isbn) == True, "Valid ISBN-13 should return True."
    
    # Invalid input test
    assert is_isbn_13(invalid_isbn) == False, "Invalid ISBN-13 should return False."
    
    # Empty input test that should fail on the mutant
    assert is_isbn_13(empty_isbn) == False, "Empty string should not be considered a valid ISBN-13."