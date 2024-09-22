from string_utils.validation import is_isbn_10


def test_is_isbn_10_detection():
    # Edge case ISBN-10 that is known to be valid
    valid_isbn_10 = '1506715214'
    # Edge case that will cause ValueError (input with non-digit characters)
    invalid_isbn_10 = '150-671521Z'  # Invalid due to non-numeric character

    # Assert valid ISBN-10 should return True
    assert is_isbn_10(valid_isbn_10) == True, f"{valid_isbn_10} should be valid."
    
    # Assert invalid ISBN-10 should return False
    assert is_isbn_10(invalid_isbn_10) == False, f"{invalid_isbn_10} should be invalid."