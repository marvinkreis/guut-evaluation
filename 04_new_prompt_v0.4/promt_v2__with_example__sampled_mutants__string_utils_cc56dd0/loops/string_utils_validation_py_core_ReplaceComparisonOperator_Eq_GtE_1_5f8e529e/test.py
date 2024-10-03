from string_utils.validation import is_isbn_13

def test_is_isbn_13_mutant_killing():
    """
    Test the is_isbn_13 function with a valid ISBN-13 number. The baseline should return True,
    while the mutant will return False due to an incorrect checksum calculation.
    """
    output = is_isbn_13('9780312498580')
    assert output == True, f"Expected True, got {output}"