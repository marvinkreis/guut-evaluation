from string_utils.manipulation import prettify

def test__saxon_genitive_formatting():
    """
    Test whether the prettify function correctly formats a Saxon genitive.
    The input 'Dave' s dog' should become 'Dave's dog' in the baseline,
    but will cause a TypeError in the mutant due to an invalid operation.
    """
    output = prettify("Dave' s dog")
    assert output == "Dave's dog", f"Unexpected output: {output}"