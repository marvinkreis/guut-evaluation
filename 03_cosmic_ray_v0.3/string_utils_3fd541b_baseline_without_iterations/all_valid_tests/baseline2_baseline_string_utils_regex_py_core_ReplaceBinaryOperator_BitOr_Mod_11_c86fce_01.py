from string_utils._regex import PRETTIFY_RE

def test__spaces_inside_regex():
    # Try to compile the 'SPACES_INSIDE' regex pattern
    try:
        # This will raise an error if the mutant is present
        assert PRETTIFY_RE['SPACES_INSIDE'] is not None
    except Exception as e:
        assert str(e) == "expected string or bytes-like object" or "Error compiling regex" in str(e)

# Execute the test
test__spaces_inside_regex()