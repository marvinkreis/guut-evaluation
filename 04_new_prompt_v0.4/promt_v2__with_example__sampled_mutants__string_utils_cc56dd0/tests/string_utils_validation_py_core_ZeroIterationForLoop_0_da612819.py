from string_utils.validation import is_isbn_13

def test_is_isbn_13_mutant_killing():
    """
    Test the is_isbn_13 function using a string with non-numeric characters. 
    The mutant will incorrectly process the input and return True, while 
    the baseline will return False for invalid ISBN-13.
    """
    output = is_isbn_13('978ABC2498580')
    assert output == False, f"Expected False, got {output}"