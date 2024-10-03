from string_utils.manipulation import prettify

def test_prettify_mutant_killing():
    """
    This test checks the prettification of a string that needs correct capitalization
    after punctuation. The input 'hello. this is a test.' should be transformed into 
    'Hello. This is a test.' in the baseline implementation. The mutant is expected to 
    incorrectly format the string, thus failing this test.
    """
    output = prettify('hello. this is a test.')
    assert output == 'Hello. This is a test.', f"Expected 'Hello. This is a test.', got {output}"