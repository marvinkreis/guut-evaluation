from string_utils.validation import is_camel_case

def test__is_camel_case():
    # This should return true for a camel case string
    assert is_camel_case('MyCamelCaseString') == True
    
    # This should return false for a regular string
    assert is_camel_case('my_camel_case_string') == False
    
    # This should return false for an empty string
    assert is_camel_case('') == False
    
    # This should return false for a number starting string
    assert is_camel_case('1stPlace') == False
    
    # This should return false for a string with special characters
    assert is_camel_case('MyString!') == False