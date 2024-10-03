from string_utils.validation import is_isbn_10

def test__is_isbn_10_mutant_killer():
    """
    Test whether the is_isbn_10 function correctly validates ISBN-10 numbers. 
    This test uses one valid and one invalid ISBN-10 number to differentiate between 
    the correct implementation (baseline) and the mutant.
    The valid input should return True, while the invalid input should return False.
    The mutant should fail this test due to its incorrect validity check.
    """
    
    valid_isbn = '1506715214'  # Known valid ISBN-10
    invalid_isbn = '1506715215'  # Known invalid ISBN-10

    assert is_isbn_10(valid_isbn) == True, "Expected True for valid ISBN-10"
    assert is_isbn_10(invalid_isbn) == False, "Expected False for invalid ISBN-10"