from string_utils.validation import is_isbn_13

def test_valid_isbn_with_hyphens_mutant_killing():
    """
    Test the is_isbn_13 function using a valid ISBN-13 number with hyphens. 
    The baseline will validate it correctly and return True,
    while the mutant will return False due to incorrect weight calculation.
    """
    valid_isbn_with_hyphens = '978-0-306-40615-7'
    output = is_isbn_13(valid_isbn_with_hyphens)
    assert output == True, f"Expected True, got {output}"