from string_utils.manipulation import prettify

def test__prettify():
    """
    Test the correct capitalization of the first letter after a punctuation mark. The input string contains several 
    punctuation marks and should have the first letter after each punctuation capitalized. The mutant modifies the 
    logic in __uppercase_first_letter_after_sign, which will cause the function to fail in this scenario.
    """
    input_string = 'hello. this is a test! is it working? yes.'
    output = prettify(input_string)
    assert output == 'Hello. This is a test! Is it working? Yes.'