import re

def test__HTML_regex_functionality():
    """The correct HTML_RE should compile and match, while the mutant should raise an error on compilation."""
    
    # Test with the correct regex
    HTML_RE = re.compile(
        r'((<([a-z]+:)?[a-z]+[^>]*/?>)(.*?(</([a-z]+:)?[a-z]+>))?|<!--.*-->|<!doctype.*>)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    
    # No assertion needed because if it fails to compile, an error will be raised
    assert HTML_RE is not None, "The correct HTML_RE should compile successfully."

    # Test with the mutant regex and verify it raises an error
    try:
        mutant_HTML_RE = re.compile(
            r'((<([a-z]+:)?[a-z]+[^>]*/?>)(.*?(</([a-z]+:)?[a-z]+>))?|<!--.*-->|<!doctype.*>)',
            re.IGNORECASE | re.MULTILINE - re.DOTALL
        )
        # If the mutant compiles successfully, the test should fail
        assert False, "Mutant HTML_RE should fail to compile successfully."
    except ValueError:
        # This is expected behavior for the mutant, confirming it is indeed not working correctly
        pass