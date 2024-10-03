from string_utils.manipulation import prettify

def test__prettify_for_invalid_input():
    """
    This test checks how the 'prettify' function handles an input that does not contain
    sufficient matching groups for the regex, which should trigger an IndexError in the mutant,
    while the baseline should handle it without exceptions.
    """
    try:
        prettify('?? this is a test!')
        assert False, "Baseline should not raise an exception."
    except IndexError:
        assert False, "Baseline should not raise an IndexError."
    except Exception:
        # Any other exception is acceptable and indicates the baseline's handling of unexpected input.
        pass

    # Mutant behavior
    try:
        prettify('?? this is a test!')
    except IndexError as e:
        print(f"Mutant correctly raised an IndexError: {str(e)}")
    except Exception as e:
        assert False, f"Mutant should only raise IndexError, but raised: {str(e)}"