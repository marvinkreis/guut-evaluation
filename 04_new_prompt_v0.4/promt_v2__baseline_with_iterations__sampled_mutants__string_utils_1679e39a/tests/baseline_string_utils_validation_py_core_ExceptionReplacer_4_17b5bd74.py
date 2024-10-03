from string_utils.validation import is_json

def test__is_json():
    """
    Test whether a string that is not valid JSON is correctly identified as such.
    The input is a malformed JSON string ('{nope}'). This test ensures that the function
    raises a ValueError during the parsing of non-JSON input. The mutant changes the exception
    handling, which would lead to an incorrect result, causing the test to fail when run against it.
    """
    output = is_json('{nope}')
    assert output == False