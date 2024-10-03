from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """
    Test whether the input '978-3-16-148410-0' is correctly identified as a valid ISBN-13 by the baseline.
    The mutant will fail to correctly calculate the product due to the altered weight calculation,
    which would lead to a return value of False for this valid ISBN-13.
    """
    output = is_isbn_13('978-3-16-148410-0')  # Valid ISBN-13 with hyphens
    assert output is True  # Expecting True for the correct implementation