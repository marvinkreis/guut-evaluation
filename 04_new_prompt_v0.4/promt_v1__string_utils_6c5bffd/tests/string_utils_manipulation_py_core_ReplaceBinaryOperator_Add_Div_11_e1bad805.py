from string_utils.manipulation import prettify

def test__prettify_with_saxon_genitive():
    """
    Test the prettify function with a saxon genitive input. The mutant should raise a TypeError
    due to the incorrect modification in the __fix_saxon_genitive method, while the baseline should
    return the correctly formatted string.
    """
    baseline_output = prettify("Dave' s dog")
    assert baseline_output == "Dave's dog", f"Expected 'Dave's dog', got '{baseline_output}'"