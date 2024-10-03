from string_utils._regex import PRETTIFY_RE

def test__saxon_genitive():
    """
    Test to check the correct compilation of the SAXON_GENITIVE regex.
    The baseline should compile without errors, while the mutant raises a TypeError
    due to an invalid operator usage in the regex definition.
    """
    try:
        saxon_genitive_regex = PRETTIFY_RE['SAXON_GENITIVE']
        assert True  # This line is reached only if compilation was successful
    except Exception as e:
        assert False, f"Exception occurred: {e}"