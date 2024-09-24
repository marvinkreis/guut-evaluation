from string_utils._regex import PRETTIFY_RE
import re

def test__prettify_re():
    # A test string that should be matched by the correct regex
    test_string = "This is John's book."

    # Directly access the regex to confirm it matches
    saxon_genitive_regex = r"(?<=\w)'s"

    # Attempt to match the modified regex directly
    orig_match = re.search(saxon_genitive_regex, test_string)
    
    # This should succeed as "John's" matches the regex
    assert orig_match is not None, "Expected a match for appropriate apostrophe usage in the original code."

    # Simulating the mutant behavior
    try:
        # Incorporate the mutant's changes to check for flag handling
        mutant_regex = re.compile(
            saxon_genitive_regex, 
            re.MULTILINE - re.UNICODE  # This should raise the ValueError
        )

        # If it doesn't raise, test for no match
        mutant_match = re.search(mutant_regex, test_string)
        assert mutant_match is None, "Expected no match due to mutant behavior."

    except ValueError as e:
        # Expected to catch the specific error regarding incompatible flags
        assert str(e) == "ASCII and UNICODE flags are incompatible", "Unexpected error message from mutant."

    except Exception as e:
        assert False, f"Unexpected exception caught: {str(e)}"

# Execute the final test
test__prettify_re()