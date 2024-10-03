from string_utils.validation import is_isbn_13

def test__is_isbn_13_mutant_killing():
    """
    This test checks the detection of an invalid ISBN-13 number. The input '9780312498581' is known 
    to be invalid and should return False in the Baseline while returning True in the Mutant.
    This ensures that the mutant is not equivalent to the Baseline.
    """
    output = is_isbn_13('9780312498581')
    assert output == False