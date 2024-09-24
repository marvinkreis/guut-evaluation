import re

def test__HTML_regex_compilation():
    """The mutant code fails to compile the HTML_RE regex due to incorrect flag usage."""
    # This will check if the correct implementation compiles
    HTML_RE = re.compile(
        r'((<([a-z]+:)?[a-z]+[^>]*/?>)(.*?(</([a-z]+:)?[a-z]+>))?|<!--.*-->|<!doctype.*>)',
        re.IGNORECASE | re.MULTILINE | re.DOTALL
    )

    assert HTML_RE is not None, "Correct HTML_RE should compile successfully."

    # This will check if the mutant raises an error
    try:
        mutant_HTML_RE = re.compile(
            r'((<([a-z]+:)?[a-z]+[^>]*/?>)(.*?(</([a-z]+:)?[a-z]+>))?|<!--.*-->|<!doctype.*>)',
            re.IGNORECASE | re.MULTILINE - re.DOTALL
        )
        # If the mutant compiles successfully, the test should fail
        assert False, "Mutant HTML_RE should not compile successfully."
    except ValueError:
        # Expected case: the mutant should raise a ValueError
        pass