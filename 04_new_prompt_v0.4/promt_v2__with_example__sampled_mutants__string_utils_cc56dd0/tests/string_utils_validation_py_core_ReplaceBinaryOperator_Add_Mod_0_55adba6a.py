from string_utils.validation import is_isbn_10

def test_is_isbn_10_mutant_killing():
    """
    Test the is_isbn_10 function with an invalid ISBN-10 number.
    The mutant will incorrectly accept the invalid ISBN-10, while the baseline will reject it.
    """
    output = is_isbn_10('1234567890')
    assert output == False, f"Expected False, got {output}"