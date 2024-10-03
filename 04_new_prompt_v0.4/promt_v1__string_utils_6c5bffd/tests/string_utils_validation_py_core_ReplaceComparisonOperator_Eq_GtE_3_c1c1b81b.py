from string_utils.validation import is_isbn_10

def test__is_isbn_10_mutant_killing():
    """
    This test case checks if the is_isbn_10 function correctly identifies invalid ISBN-10 numbers.
    Specifically, it verifies that an input longer than 10 characters does not validate as a correct ISBN-10.
    The original (baseline) implementation should return False, while the mutant will incorrectly return True.
    """
    isbn_invalid = '15067152145'  # Invalid ISBN-10
    assert is_isbn_10(isbn_invalid) == False