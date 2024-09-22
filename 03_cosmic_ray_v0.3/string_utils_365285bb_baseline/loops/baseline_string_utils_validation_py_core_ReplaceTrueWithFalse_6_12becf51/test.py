from string_utils.validation import is_isbn

def test_is_isbn():
    # Test case for valid ISBN 10 with normalization
    assert is_isbn('1506715214') == True
    # Test case for valid ISBN 13 with normalization
    assert is_isbn('9780312498580') == True
    # Test case for valid ISBN 10 with hyphens
    assert is_isbn('150-6715214') == True
    # Test case for valid ISBN 13 with hyphens
    assert is_isbn('978-0312498580') == True
    # Test case with normalization set to False (should fail in the mutant)
    assert is_isbn('150-6715214', normalize=False) == False  # This should fail in the mutant

    # Invalid cases
    assert is_isbn('invalid_isbn') == False
    assert is_isbn('1234567890123') == False  # Invalid ISBN