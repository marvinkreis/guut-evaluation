from string_utils.validation import is_isbn_13

def test__is_isbn_13_invalid_long_input():
    """
    This test checks if the ISBN validation system correctly identifies that a 14-character string 
    is not a valid ISBN-13 number. The baseline should return False, while the mutant should return True.
    """
    long_isbn = '97803124985800'  # This input is 14 characters long
    output = is_isbn_13(long_isbn)
    print(f"output = {output}")
    assert output is False  # Expect the baseline to return False