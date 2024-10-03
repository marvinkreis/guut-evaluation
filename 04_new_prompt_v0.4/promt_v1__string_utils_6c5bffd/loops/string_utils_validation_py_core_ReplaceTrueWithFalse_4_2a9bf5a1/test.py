from string_utils.validation import is_isbn_10

def test_is_isbn_10_mutant_killer():
    """
    Test the `is_isbn_10` function to distinguish between the baseline and the mutant.
    The input '150-6715214' should return true in the baseline but false in the mutant
    due to the change in the normalize parameter default value.
    """
    hyphenated_isbn = '150-6715214'  # expected True for baseline, False for mutant
    result = is_isbn_10(hyphenated_isbn)
    
    assert result == True, f"Expected True but got {result} (This should pass for baseline)"