from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    # Valid ISBN-13 with 13 digits
    valid_isbn_13 = '9780312498580'  # This should return True
    assert is_isbn_13(valid_isbn_13) == True, f"Failed on valid ISBN-13: {valid_isbn_13}"
    
    # Invalid ISBN-13 with 14 digits (should not be a valid length)
    invalid_isbn_13 = '97803124985800'  # This should return False
    assert is_isbn_13(invalid_isbn_13) == False, f"Failed on invalid ISBN-13: {invalid_isbn_13}"
    
    # Another valid ISBN-13 variant with hyphens
    valid_isbn_13_hyphenated = '978-0312498580'  # This should return True
    assert is_isbn_13(valid_isbn_13_hyphenated) == True, f"Failed on valid ISBN-13 with hyphens: {valid_isbn_13_hyphenated}"
    
    # Invalid ISBN-13 example with wrong length
    invalid_isbn_13_short = '97803124985'  # This should return False
    assert is_isbn_13(invalid_isbn_13_short) == False, f"Failed on short ISBN-13: {invalid_isbn_13_short}"

    # Empty string should return False
    assert is_isbn_13('') == False, "Failed on empty string"

    print("All tests passed.")