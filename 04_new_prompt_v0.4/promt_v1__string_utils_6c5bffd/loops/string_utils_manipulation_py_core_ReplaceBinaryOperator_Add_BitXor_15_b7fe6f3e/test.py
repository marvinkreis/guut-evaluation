from string_utils.manipulation import slugify

def test__slugify_with_consecutive_separators():
    """
    Test the slugify function with consecutive separators. 
    The baseline implementation will normalize consecutive separators, returning a properly formatted slug. 
    The mutant implementation will fail due to a TypeError caused by an invalid operator usage.
    """
    input_string = 'Top---10---Reasons---To---Love---Dogs!!!'
    output = slugify(input_string, separator='-')
    assert output == 'top-10-reasons-to-love-dogs'