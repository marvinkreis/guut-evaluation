from string_utils.validation import is_email

def test__is_email():
    """
    Test the email validation to check whether the mutant correctly handles the 
    logic changed from 'or' to 'and' for email tail length.
    The input is a valid head but invalid tail which should return True in baseline 
    and False in mutant.
    """
    test_email = 'a' * 65 + '@example.com'  # head is 65 characters, invalid
    output = is_email(test_email)
    assert output == False  # Expecting False since the tail exceeds the limit.