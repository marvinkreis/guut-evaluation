from string_utils.validation import is_email

def test__is_email():
    """
    This test checks the email validation function. A well-formed email 
    address should return True, while an incorrect format should return False. 
    The test demonstrates the mutant's failure due to an invalid operation 
    when trying to construct the email string in the mutant implementation.
    """
    valid_email = "my.email@the-provider.com"
    invalid_email = "@gmail.com"
    
    # Test for valid email
    output_valid = is_email(valid_email)
    assert output_valid == True, f"Expected True for valid email, got {output_valid}"
    
    # Test for invalid email
    output_invalid = is_email(invalid_email)
    assert output_invalid == False, f"Expected False for invalid email, got {output_invalid}"