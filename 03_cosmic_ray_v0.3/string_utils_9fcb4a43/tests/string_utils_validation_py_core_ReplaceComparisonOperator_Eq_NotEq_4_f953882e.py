from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    """The mutant changes the condition from `product % 11 == 0` to `product % 11 != 0`, 
       which invalidates valid ISBN-10 numbers.
    """
    valid_isbn_numbers = ['1506715214', '0-306-40615-2', '0-19-852663-6']
    for isbn in valid_isbn_numbers:
        result = is_isbn_10(isbn)
        assert result is True, f"{isbn} should be valid but was reported as invalid"