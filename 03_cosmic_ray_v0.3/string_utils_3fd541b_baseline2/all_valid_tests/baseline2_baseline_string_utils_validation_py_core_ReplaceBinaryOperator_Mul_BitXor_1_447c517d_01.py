from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # Testing valid ISBN-10
    valid_isbn_10 = '1506715214'  # Valid ISBN-10
    assert is_isbn_10(valid_isbn_10) == True, f"Expected True for {valid_isbn_10}"
    
    # Testing invalid ISBN-10
    invalid_isbn_10 = '1506715215'  # Invalid ISBN-10 (checksum does not match)
    assert is_isbn_10(invalid_isbn_10) == False, f"Expected False for {invalid_isbn_10}"
    
    # Testing cases with hyphens
    valid_isbn_with_hyphen = '150-6715214'  # Valid ISBN-10
    assert is_isbn_10(valid_isbn_with_hyphen) == True, f"Expected True for {valid_isbn_with_hyphen}"
    
    invalid_isbn_with_hyphen = '150-6715215'  # Invalid ISBN-10
    assert is_isbn_10(invalid_isbn_with_hyphen) == False, f"Expected False for {invalid_isbn_with_hyphen}"
    
    print("All tests passed.")