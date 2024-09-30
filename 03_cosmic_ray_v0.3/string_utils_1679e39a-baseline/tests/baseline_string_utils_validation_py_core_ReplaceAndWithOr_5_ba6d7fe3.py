from string_utils.validation import is_camel_case

def test__is_camel_case():
    """
    Test if a string is recognized as camel case. The input 'MyString' is a valid camel case string,
    and under the baseline implementation would return True. However, with the mutant's change from 
    `and` to `or`, it would incorrectly return True for non-camel case strings such as 'mystring'. 
    This test checks that the function correctly validates 'MyString'.
    """
    output = is_camel_case('MyString')
    assert output is True

def test__is_not_camel_case():
    """
    Test if an invalid camel case string returns False. The input 'mystring' is not a valid camel case string,
    and it should return False under the baseline implementation. However, with the mutant's change, 
    it may return True incorrectly.
    """
    output = is_camel_case('mystring')
    assert output is False