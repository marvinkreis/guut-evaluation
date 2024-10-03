from string_utils.validation import is_isbn_13

def test__isbn_empty_string():
    """
    Testing the behavior of the ISBN validation function with an empty string,
    which should return False in the baseline but True in the mutant.
    This confirms the mutant's failure when accepting empty strings as valid ISBNs.
    """
    output = is_isbn_13('')
    assert output == False  # Expected to pass for the baseline and fail for the mutant.