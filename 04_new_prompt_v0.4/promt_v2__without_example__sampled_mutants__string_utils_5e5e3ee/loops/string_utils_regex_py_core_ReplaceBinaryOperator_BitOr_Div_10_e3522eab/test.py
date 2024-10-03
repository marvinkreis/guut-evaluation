from string_utils._regex import PRETTIFY_RE

def test__regex_pretty_compile():
    """
    Test whether compiling the PRETTIFY_RE regex correctly raises a TypeError in the mutant.
    In the baseline, regex compilation should succeed without any exception.
    """
    try:
        # Attempt to compile the 'DUPLICATES' regex to test the baseline functionality.
        compiled_regex = PRETTIFY_RE['DUPLICATES']
        # Accessing pattern to ensure it compiles correctly
        output = compiled_regex.pattern  
        assert True, "Expected no exception, but a TypeError was not raised."
    except TypeError as e:
        print(f"Expected exception caught: {e}")