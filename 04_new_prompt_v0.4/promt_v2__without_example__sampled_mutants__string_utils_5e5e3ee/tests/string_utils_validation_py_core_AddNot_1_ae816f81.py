from string_utils.validation import is_isbn_13

def test__is_isbn_13_with_hyphens():
    """
    Verify that the is_isbn_13 function correctly handles an ISBN-13 input with hyphens. 
    The baseline should return True for a valid ISBN when normalization is applied, while
    the mutant should return False due to the incorrect handling of the normalization flag.
    """
    input_string = '978-0312498580'  # Valid ISBN-13 with hyphens
    assert is_isbn_13(input_string, normalize=True) == True