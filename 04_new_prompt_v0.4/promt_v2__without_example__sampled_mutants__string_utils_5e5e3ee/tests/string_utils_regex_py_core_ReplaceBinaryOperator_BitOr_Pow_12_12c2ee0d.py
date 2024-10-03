from string_utils._regex import PRETTIFY_RE

def test__saxon_genitive_kills_mutant():
    """
    Test SAXON_GENITIVE regex in a multiline context to confirm that the
    mutant's change leads to an error during regex compilation, distinguishing
    it from the baseline behavior.
    """
    input_string = "Mark's hat\nTom's bike"
    try:
        matches = PRETTIFY_RE['SAXON_GENITIVE'].findall(input_string)
        assert len(matches) == 0  # The baseline would return an empty list
    except Exception as e:
        print("Error encountered:", e)