from string_utils.validation import is_isbn_10

def test__is_isbn_10_mutant_kill():
    """
    Test the is_isbn_10 function to confirm that the mutant fails while the baseline passes.
    The input '1506715214' is a valid ISBN-10 and should return True in the baseline but False in the mutant.
    This checks if the method works correctly without the mutated bitwise AND operation.
    """
    assert is_isbn_10('1506715214') == True  # should pass in baseline