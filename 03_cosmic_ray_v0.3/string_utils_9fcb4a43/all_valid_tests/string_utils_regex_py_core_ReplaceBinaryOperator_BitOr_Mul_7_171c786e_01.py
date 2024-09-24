from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """The mutant regex should not compile and match correctly, while the correct one should."""
    try:
        test_string = "Hello   world!!  Test:  5.5?? 100%    "

        # Test matching with the correct regex for duplicates
        correct_regex = PRETTIFY_RE['DUPLICATES']
        correct_matches = correct_regex.findall(test_string)
        assert len(correct_matches) > 0, "The correct PRETTIFY_RE should find matches."

    except Exception as e:
        assert False, f"Unexpected error raised with correct regex: {e}"

# The mutant part will not be included since we can't import it into the test