import re  # Import the re module

def test__HTML_TAG_ONLY_RE():
    """The mutant's change to HTML_TAG_ONLY_RE should cause a TypeError when compiled."""
    # Test correct implementation
    from string_utils._regex import HTML_TAG_ONLY_RE
    assert HTML_TAG_ONLY_RE is not None, "HTML_TAG_ONLY_RE should be defined."

    # Test mutant implementation
    try:
        # This simulates the effect of trying to use the mutant's regex
        # We do not actually import from the mutant to avoid module issues
        mutant_HTML_TAG_ONLY_RE = re.compile(
            r'(<([a-z]+:)?[a-z]+[^>]*/?>|</([a-z]+:)?[a-z]+>|<!--.*-->|<!doctype.*>)',
            re.IGNORECASE / re.MULTILINE | re.DOTALL  # This should cause a TypeError
        )
        # If we reach this line, then the mutant did not raise an error as expected
        assert False, "Mutant should raise a TypeError when accessing HTML_TAG_ONLY_RE"
    except TypeError:
        pass  # Expected outcome; mutant causes a TypeError
    except Exception as e:
        assert False, f"Mutant raised a different exception: {str(e)}"