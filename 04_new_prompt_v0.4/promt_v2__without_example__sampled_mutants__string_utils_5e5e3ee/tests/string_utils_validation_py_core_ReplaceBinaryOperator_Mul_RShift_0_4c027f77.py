from string_utils.validation import is_isbn_13

def test__isbn_checker_mutant_killing():
    """
    This test checks for a valid ISBN-13 number. The input '9780312498580' is a valid ISBN-13,
    and should return true. The mutant implementation will return false due to the faulty
    adjustment of the multiplication operation to a bitwise operation.
    """
    assert is_isbn_13('9780312498580') == True