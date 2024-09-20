from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """The mutant changes the default value of the normalize parameter, which causes it to return False for valid ISBN 13 with hyphens."""
    isbn_test = '978-0312498580'
    output = is_isbn_13(isbn_test)
    assert output == True, "is_isbn_13 must validate ISBN 13 strings containing hyphens"