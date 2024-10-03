from string_utils.validation import is_credit_card

def test__is_credit_card_empty_string():
    """
    Test the is_credit_card function with an empty string input. 
    The baseline should return False, while the mutant should return True. 
    This tests the correctness of input validation in the function.
    """
    output = is_credit_card('')
    assert output is False  # Expecting False on baseline