from string_utils.validation import is_isbn

def test_is_isbn_mutant_killing():
    """
    Test the is_isbn function with a valid ISBN-13 that contains hyphens.
    The mutant will incorrectly return False due to the normalization parameter 
    being set to False, while the baseline will return True.
    """
    output = is_isbn('978-3-16-148410-0')
    assert output == True, f"Expected True, got {output}"