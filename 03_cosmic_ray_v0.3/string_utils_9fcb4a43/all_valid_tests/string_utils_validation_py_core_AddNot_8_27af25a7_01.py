from string_utils.validation import is_email

def test__is_email():
    """The mutant changes the email validation logic, causing it to return False for valid emails."""
    
    # Valid emails
    assert is_email('example@example.com') == True, "Should validate a proper email."
    assert is_email('user.name+tag+sorting@example.com') == True, "Should validate an email with tags."
    assert is_email('user@site.co.uk') == True, "Should validate an email with country code."
    assert is_email('student_name@university.edu') == True, "Should validate an academic email."

    # Invalid emails
    assert is_email('') == False, "Empty string should not be a valid email."
    assert is_email('plainaddress') == False, "String without @ should not be valid."
    assert is_email('@missingusername.com') == False, "Email without user part should not be valid."
    assert is_email('username@.com') == False, "Email with domain starting with dot should not be valid."