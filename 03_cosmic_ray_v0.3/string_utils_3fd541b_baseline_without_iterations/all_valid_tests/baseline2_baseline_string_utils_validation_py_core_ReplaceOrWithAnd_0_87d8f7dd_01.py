from string_utils.validation import is_email

def test__is_email():
    # Test case for the original version
    result1 = is_email('my.email@the-provider.com')
    assert result1 == True  # Should return True for a valid email

    # Test case that should fail in the mutant version
    result2 = is_email('')
    assert result2 == False  # Should return False since the input is empty, which is invalid