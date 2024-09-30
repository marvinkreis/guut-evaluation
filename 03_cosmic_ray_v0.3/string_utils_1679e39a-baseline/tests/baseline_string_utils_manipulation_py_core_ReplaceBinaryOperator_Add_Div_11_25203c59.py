from string_utils.manipulation import prettify

def test__prettify():
    """
    Test for the correct handling of apostrophes and spaces in a Saxon genitive.
    The input tests whether the mutant change (using '/' instead of '+') causes an error in the output.
    The expected output is "Dave's dog", while the mutant will result in a TypeError due to invalid operation with strings.
    """
    output = prettify("Dave' s dog")
    assert output == "Dave's dog"