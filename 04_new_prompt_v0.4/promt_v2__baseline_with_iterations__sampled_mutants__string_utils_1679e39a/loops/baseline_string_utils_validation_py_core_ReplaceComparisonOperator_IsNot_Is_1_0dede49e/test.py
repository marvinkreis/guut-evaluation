from string_utils.validation import is_url

def test__is_url():
    """
    Test if the is_url function correctly identifies a valid URL. 
    The input is a well-formed URL ('http://www.example.com'), which should return True 
    with the correct implementation. The mutant inverts the match condition and will return 
    False for this URL, thus allowing us to detect the mutant.
    """
    output = is_url('http://www.example.com')
    assert output == True