from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    """The mutant allows a string longer than 10 characters to pass as a valid ISBN-10"""
    input_string = "15067152141"  # Invalid ISBN-10
    output = is_isbn_10(input_string)
    assert output is False, "is_isbn_10 should return False for invalid ISBN-10 inputs."