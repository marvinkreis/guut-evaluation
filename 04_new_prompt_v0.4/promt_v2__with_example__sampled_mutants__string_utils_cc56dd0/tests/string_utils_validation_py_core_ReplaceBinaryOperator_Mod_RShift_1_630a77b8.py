from string_utils.validation import is_isbn_13

def test_is_isbn_13_mutant_killing():
    """
    Test the is_isbn_13 function using an invalid ISBN-13 string. 
    The baseline should return False, while the mutant will return True 
    due to an incorrect check in the code.
    """
    output = is_isbn_13('1234567890123')
    assert output == False, f"Expected False, got {output}"