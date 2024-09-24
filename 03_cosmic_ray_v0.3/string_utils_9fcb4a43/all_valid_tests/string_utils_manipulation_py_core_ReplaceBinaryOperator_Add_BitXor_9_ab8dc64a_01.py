from string_utils.manipulation import prettify

def test__prettify():
    """
    The mutant replaces '+' with '^' in __ensure_spaces_around, 
    which leads to incorrect formatting in some input scenarios.
    """
    output = prettify(' unprettified string , like this one!     ')
    assert output == 'Unprettified string, like this one!', "Expected space normalization and appending punctuation rules."
    
    edge_case_output = prettify('This is a test.  This should be normalized!   ')
    assert edge_case_output == 'This is a test. This should be normalized!', "Expected formatting around punctuation to be applied."