from string_utils.validation import is_isbn

def test__is_isbn():
    # Test case to detect the mutant
    valid_isbn_13 = '9780312498580'  # This is a valid ISBN-13
    valid_isbn_10 = '1506715214'      # This is a valid ISBN-10

    # The original implementation should return True for both valid ISBNs with normalize=True
    assert is_isbn(valid_isbn_13, normalize=True) == True
    assert is_isbn(valid_isbn_10, normalize=True) == True
  
    # The mutant implementation will fail this test because it sets normalize=False by default,
    # and will ignore valid inputs while checking, as the normalize flag is important for proper validation.
    assert is_isbn(valid_isbn_13) == True       # This should work as expected
    assert is_isbn(valid_isbn_10) == True       # This should work as expected
    
    invalid_isbn_10 = '150-6715215'  # Not a valid ISBN-10
    invalid_isbn_13 = '978-0312498581'  # Not a valid ISBN-13

    # Testing with an invalid ISBN which should definitely return False
    assert is_isbn(invalid_isbn_10) == False   # This should work as expected
    assert is_isbn(invalid_isbn_13) == False   # This should work as expected
    
    # Moreover we can check for a scenario with normalization
    assert is_isbn('978-0-306-40615-7', normalize=True) == True  # Valid ISBN-13 with hyphens
    assert is_isbn('0-306-40615-2', normalize=True) == True      # Valid ISBN-10 with hyphens

    print("All assertions passed!")