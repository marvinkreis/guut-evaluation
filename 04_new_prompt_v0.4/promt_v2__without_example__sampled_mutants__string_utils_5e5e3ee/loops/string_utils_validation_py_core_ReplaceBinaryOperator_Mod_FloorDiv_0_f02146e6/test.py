from string_utils.validation import is_isbn_13

def test__is_isbn_13_mutant_killing():
    """
    Test the ISBN-13 validation function with a valid ISBN-13 number.
    The input '9780470059029' should return True for the baseline,
    while the mutant is expected to return False due to the altered weight calculation.
    This test should successfully identify the mutant.
    """
    output = is_isbn_13('9780470059029')
    assert output == True  # Expecting the output to be True for the correct ISBN-13