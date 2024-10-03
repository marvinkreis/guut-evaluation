from string_utils.validation import is_isbn_13

def test__is_isbn_13_kill_mutant():
    """
    Test the validity of a valid ISBN-13 number. The input '978-0312498580' is a known valid ISBN-13. 
    The mutant should return False due to the weight change, while the baseline should return True.
    This test confirms that the change in weight calculation in the mutant causes it to fail where the baseline succeeds.
    """
    valid_isbn = '978-0312498580'
    output = is_isbn_13(valid_isbn)
    assert output == True, f"Expected True but got {output}"