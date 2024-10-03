from string_utils.manipulation import prettify

def test_prettify_mutant_killing():
    """
    Test the prettify function with complex spacing and punctuation.
    The baseline will correctly format the string, while the mutant will 
    raise a TypeError due to status logic that attempts to subtract a string.
    """
    output = prettify('   This is    an example;   with  irregular spacing    .   ')
    assert output == 'This is an example; with irregular spacing.', f"Expected 'This is an example; with irregular spacing.', got '{output}'"