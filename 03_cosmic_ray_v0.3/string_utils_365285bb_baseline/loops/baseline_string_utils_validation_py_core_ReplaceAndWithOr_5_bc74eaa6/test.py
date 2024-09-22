from string_utils.validation import is_camel_case

def test_is_camel_case():
    # Test with a valid camel case string
    assert is_camel_case('MyString') == True  # Should return True
    # Test with an invalid camel case string
    assert is_camel_case('mystring') == False  # Should return False
    
    # This input is empty, which in the original logic should return False
    assert is_camel_case('') == False  # Should return False in original, True in mutant
    # Another variant with an underscore, which should also return False
    assert is_camel_case('my_string') == False  # Should return False in original, True in mutant