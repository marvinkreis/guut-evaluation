from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    """
    This test checks the is_isbn_10 function with a specific invalid input '1506715215'. The baseline implementation
    should return False for this incorrect ISBN-10, since it does not validate as a valid ISBN-10. The mutant, however,
    will incorrectly evaluate this due to its wrong bitwise operation, so it will return True for this invalid input,
    which will fail the test. This ensures that the mutant is detected.
    """
    output = is_isbn_10('1506715215')  # Invalid ISBN-10
    assert output == False  # Expected to return False for invalid ISBN-10