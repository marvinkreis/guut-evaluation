from string_utils.validation import is_isbn

def test__is_isbn():
    """The mutant changes 'or' to 'and', which causes valid ISBN-10 inputs to fail."""
    valid_isbn10 = '1506715214'  # A valid ISBN-10
    assert is_isbn(valid_isbn10) == True, "is_isbn should return True for valid ISBN-10"