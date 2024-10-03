from string_utils.validation import is_isbn_13

def test__isbn_mutant_killer():
    """
    This test checks if the function is_isbn_13 can handle inputs that are not valid ISBN-13 formats. 
    The input 'abcdefghijklm' should return False in the baseline implementation but True in the mutant 
    due to the change made in the return statement of the is_isbn_13 method.
    """
    invalid_format_isbn = 'abcdefghijklm'  # A string of 13 characters that are not valid ISBN-13

    output = is_isbn_13(invalid_format_isbn)
    assert output == False  # This should fail on the mutant