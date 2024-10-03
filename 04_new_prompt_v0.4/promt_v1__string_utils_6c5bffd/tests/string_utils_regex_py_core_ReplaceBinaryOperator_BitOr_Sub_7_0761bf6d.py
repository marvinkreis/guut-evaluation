from string_utils._regex import PRETTIFY_RE

def test__prettify_re_invalid_flag():
    """
    This test checks whether the PRETTIFY_RE regex compilation raises an error due to incompatible flags in the mutant. 
    The baseline should compile successfully, while the mutant should raise a ValueError.
    """
    try:
        # Attempt to access the 'DUPLICATES' regex
        PRETTIFY_RE['DUPLICATES']
        assert True  # The baseline should reach this point without error
    except ValueError as ex:
        print(f"ValueError encountered in mutant: {ex}")
        assert False  # The mutant should raise an error here

test__prettify_re_invalid_flag()