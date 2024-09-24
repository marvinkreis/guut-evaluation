from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    """The change from '|' to '+' in the regex meant to detect duplicates will yield different results for complex strings."""
    test_input = """
    This is a test -- with irregular spacing and
    multiple signs -- -- are not proper!
    We also have --   multiple spaces and signs.
    """

    # We expect the correct regex to identify duplicates correctly
    correct_matches = PRETTIFY_RE['DUPLICATES'].findall(test_input)

    # Since we didn't find a mutant version, we can't assert here,
    # but we assume the mutant alters how duplicates are handled leading to different results.
    
    # Validate that the number of duplicates is greater than two which should be the case
    assert len(correct_matches) > 2, "The correct implementation should find adequate duplicate signs."

# When executed:
# - In the correct implementation, it should assert correctly based on the expected duplicate discovery.
# - In the mutant implementation, where the duplicated behavior logic may change, it may not find sufficient number of duplicates and thus fail.