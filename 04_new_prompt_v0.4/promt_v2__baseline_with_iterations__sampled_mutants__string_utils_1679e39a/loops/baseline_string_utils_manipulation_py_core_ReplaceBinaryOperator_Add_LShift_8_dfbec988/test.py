from string_utils.manipulation import prettify

def test__prettify():
    """
    Test whether the prettify formatting correctly adds spaces around a sentence.
    The input string 'unprettified string,,like this one,will be"prettified".' 
    should be reformatted to include appropriate spaces, and will fail with the mutant 
    since it erroneously uses '<<' instead of '+' to concatenate strings,
    resulting in an invalid output.
    """
    output = prettify('unprettified string,,like this one,will be"prettified".')
    assert output == 'Unprettified string, like this one, will be "prettified".'