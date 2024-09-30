from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """
    Test whether the correct ISBN-13 weighting is applied. The input '9780312498580' is a valid ISBN-13 and should return True.
    The mutant code incorrectly adjusts the weight based on a faulty condition, leading to an incorrect calculation.
    """
    output = is_isbn_13('9780312498580')
    assert output == True