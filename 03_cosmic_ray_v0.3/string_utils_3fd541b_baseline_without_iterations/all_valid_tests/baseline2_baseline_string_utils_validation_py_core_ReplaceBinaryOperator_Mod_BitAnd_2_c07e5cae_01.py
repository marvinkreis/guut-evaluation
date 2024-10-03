from string_utils.validation import is_isbn_10

def test__is_isbn_10():
    # This test case is designed to detect the difference between the original 
    # function and the mutant where the operator was changed from '%' to '&'.
    
    # A valid ISBN-10 example
    valid_isbn_10 = '1506715214'  # This should return True.
    assert is_isbn_10(valid_isbn_10) == True, "The valid ISBN-10 should return True."
    
    # A test case that would not have been affected by the mutant change
    invalid_isbn_10 = '1506715213'  # This should return False.
    assert is_isbn_10(invalid_isbn_10) == False, "The invalid ISBN-10 should return False."
    
    # This case should pass in the original implementation but trigger the mutant
    incorrect_isbn_10 = '150-6715214'  # Normally valid, should equal True, but mutant changes outcome.
    assert is_isbn_10(incorrect_isbn_10) == True, "The ISBN-10 with hyphens should return True."