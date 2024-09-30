from string_utils.validation import is_email

def test__is_email():
    """
    Test an invalid email case where the local part exceeds the maximum allowed length.
    The input 'a' * 65 + '@example.com' should return false with the baseline, 
    but the mutant incorrectly allows this by checking for a length of > 65, resulting in true.
    """
    output = is_email('a' * 65 + '@example.com')
    assert output is False