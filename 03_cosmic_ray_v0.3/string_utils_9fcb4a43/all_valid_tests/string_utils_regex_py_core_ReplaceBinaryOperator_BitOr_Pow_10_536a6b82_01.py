from string_utils._regex import PRETTIFY_RE

def test__prettify_regex_access():
    """Test that accessing PRETTIFY_RE does not raise errors."""
    try:
        # Check to see if we can access PRETTIFY_RE without issue.
        assert PRETTIFY_RE is not None, "PRETTIFY_RE should exist and be accessible."
        print("Accessed PRETTIFY_RE successfully.")
    except Exception as e:
        # Any exception would imply issues with the mutant.
        assert False, f"Failed to access PRETTIFY_RE: {e}"

# Running the test function
test__prettify_regex_access()