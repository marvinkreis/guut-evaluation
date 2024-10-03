from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE_saxon_genitive_compilation():
    """
    Test that accessing the SAXON_GENITIVE regex will compile successfully in the baseline,
    but will raise a ValueError in the mutant due to altering the UNICODE flag to be negative.
    """
    try:
        # Attempt to compile SAXON_GENITIVE regex. This should work in baseline.
        regex = PRETTIFY_RE['SAXON_GENITIVE']
        assert regex  # This will pass if the regex is accessible and was compiled successfully.
    except ValueError:
        assert False, "ValueError encountered in baseline code; it should compile successfully."