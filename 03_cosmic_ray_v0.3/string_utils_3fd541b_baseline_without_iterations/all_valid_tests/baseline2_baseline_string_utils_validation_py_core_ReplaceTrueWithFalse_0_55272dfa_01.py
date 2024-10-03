from string_utils.validation import is_isbn_10, is_isbn_13

def test__is_isbn():
    # Test that ISBN-10 is correctly validated with normalize=True
    assert is_isbn_10('150-6715214') == True
    assert is_isbn_10('1506715214') == True
    
    # Test that ISBN-10 is correctly rejected with normalize=False
    assert is_isbn_10('150-6715214', normalize=False) == False
    
    # Test that ISBN-13 is correctly validated with normalize=True
    assert is_isbn_13('978-0312498580') == True
    assert is_isbn_13('9780312498580') == True
    
    # Test that ISBN-13 is correctly rejected with normalize=False
    assert is_isbn_13('978-0312498580', normalize=False) == False

    # Edge case with an invalid ISBN
    assert is_isbn_10('1234567890') == False  # Assuming it's invalid ISBN-10
    assert is_isbn_13('1234567890123') == False  # Assuming it's invalid ISBN-13