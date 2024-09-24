from string_utils.validation import is_isbn_13

def test_is_isbn_13():
    # Test with a valid ISBN-13 that should pass with normalize=True
    valid_isbn_13 = '978-0312498580'
    assert is_isbn_13(valid_isbn_13) == True  # This should return True with correct code

    # Test with a valid ISBN-13 that should fail with normalize=False
    assert is_isbn_13(valid_isbn_13, normalize=False) == False  # This will fail with the mutant