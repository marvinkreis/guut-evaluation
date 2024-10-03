from string_utils._regex import PRETTIFY_RE

def test__prettify_re_compile():
    try:
        # Attempt to compile PRETTIFY_RE
        compiled_regex = PRETTIFY_RE['RIGHT_SPACE']
        compiled_regex.pattern  # Accessing the pattern to ensure it compiles correctly
        assert True  # If we reached here, compilation was successful.
    except Exception as e:
        # If an exception was raised, compilation failed
        assert False, f"Regex compilation failed with exception: {e}"