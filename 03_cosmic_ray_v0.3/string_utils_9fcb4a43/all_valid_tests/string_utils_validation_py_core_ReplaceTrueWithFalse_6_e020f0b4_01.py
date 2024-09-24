from string_utils.validation import is_isbn

def test__is_isbn():
    """Testing is_isbn function; check behavior with normalization defaults."""

    # Valid ISBN inputs (expected to pass for both implementations)
    valid_isbn_10 = "1506715214"  # Valid ISBN 10 without hyphens
    valid_isbn_13 = "9780312498580"  # Valid ISBN 13 without hyphens

    # Valid ISBN should return True
    assert is_isbn(valid_isbn_10) is True, "Valid ISBN 10 should be accepted."
    assert is_isbn(valid_isbn_13) is True, "Valid ISBN 13 should be accepted."

    # Invalid ISBNs (expected to fail for both)
    invalid_isbn_mixed = "150a6715214"  # Invalid due to letters
    assert not is_isbn(invalid_isbn_mixed), "Invalid ISBN with letters should be rejected."

    # Edge Cases
    edge_case_1 = "978-0-123456-47-X" # Invalid format but resembles an ISBN
    edge_case_2 = "INVALID-ISBN-123456"  # Completely invalid formatted string
    assert not is_isbn(edge_case_1), "Invalid edge case should be rejected."
    assert not is_isbn(edge_case_2), "Completely invalid edge case should be rejected."

    # Testing the mutant with various mixed scenarios where normalization plays a role
    valid_isbn_10_with_hyphen = "150-6715214"  # Valid ISBN 10 with hyphen
    assert is_isbn(valid_isbn_10_with_hyphen, normalize=False) is False, "Mutant should reject valid ISBN 10 with hyphen due to normalization being False."

    valid_isbn_13_with_hyphen = "978-0312498580"  # Valid ISBN 13 with hyphen
    assert is_isbn(valid_isbn_13_with_hyphen, normalize=False) is False, "Mutant should reject valid ISBN 13 with hyphen due to normalization being False."

# Execute the test
test__is_isbn()