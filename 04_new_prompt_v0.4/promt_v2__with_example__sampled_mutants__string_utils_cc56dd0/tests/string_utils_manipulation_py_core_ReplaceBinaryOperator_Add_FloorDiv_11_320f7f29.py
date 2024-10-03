from string_utils.manipulation import prettify

def test_prettify_mutant_killing():
    """
    Test the prettify function using a string with a Saxon genitive structure.
    The baseline should return 'Dave's dog', while the mutant should raise a 
    TypeError due to the incorrect usage of the floor division operator.
    """
    output = prettify("Dave' s dog")
    assert output == "Dave's dog", f"Expected 'Dave's dog', got {output}"