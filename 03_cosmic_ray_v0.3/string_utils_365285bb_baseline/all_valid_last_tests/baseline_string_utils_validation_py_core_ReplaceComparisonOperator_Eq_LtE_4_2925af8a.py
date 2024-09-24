from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # Valid ISBN-10 numbers
    valid_isbn_1 = '0471958697'  # Should return True
    valid_isbn_2 = '0306406152'   # Should return True
    valid_isbn_3 = '0747532745'   # Should return True

    # Check valid ISBN-10 numbers
    assert is_isbn_10(valid_isbn_1) == True, "The function should return True for valid ISBN-10."
    assert is_isbn_10(valid_isbn_2) == True, "The function should return True for valid ISBN-10."
    assert is_isbn_10(valid_isbn_3) == True, "The function should return True for valid ISBN-10."

    # Invalid ISBN-10 numbers
    invalid_isbn_1 = '0471958695'  # Invalid (fails checksum)
    invalid_isbn_2 = '1234567890'   # Invalid (not a real ISBN-10)
    invalid_isbn_3 = '1506715200'   # Should fail, confirmed invalid

    # Check invalid ISBN-10 numbers
    assert is_isbn_10(invalid_isbn_1) == False, "The function should return False for an invalid ISBN-10."
    assert is_isbn_10(invalid_isbn_2) == False, "The function should return False for an invalid ISBN-10."
    assert is_isbn_10(invalid_isbn_3) == False, "The function should return False for an invalid ISBN-10."

# To run the test function directly
test__is_isbn_10()