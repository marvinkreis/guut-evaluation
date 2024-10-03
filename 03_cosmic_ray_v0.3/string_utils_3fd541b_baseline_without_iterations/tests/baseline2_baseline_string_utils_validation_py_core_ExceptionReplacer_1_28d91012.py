from string_utils.validation import is_isbn_10

def test__is_isbn_10_invalid_input_handling():
    # Test with a string that can't be converted to an integer, which should raise ValueError
    invalid_isbn = '150-67A5214'  # Invalid due to non-numeric character 'A'
    
    # We expect this to return False, as it cannot be a valid ISBN 10
    assert is_isbn_10(invalid_isbn) == False

    # Additionally, we will also check a valid ISBN for correct behavior
    valid_isbn = '1506715214'
    assert is_isbn_10(valid_isbn) == True