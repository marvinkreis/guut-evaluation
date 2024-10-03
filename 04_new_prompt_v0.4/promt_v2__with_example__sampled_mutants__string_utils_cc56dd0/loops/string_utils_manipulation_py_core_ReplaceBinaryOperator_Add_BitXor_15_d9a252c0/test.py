from string_utils.manipulation import slugify

def test_slugify_mutant_killing():
    """
    This test checks the slugification of a string that contains consecutive separators.
    The baseline returns a properly slugified string without consecutive separators,
    while the mutant raises a TypeError due to incorrect operator usage.
    """
    output = slugify("Hello--World")
    assert output == "hello-world", f"Expected 'hello-world', got '{output}'"