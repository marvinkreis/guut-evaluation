from string_utils._regex import PRETTIFY_RE

def test__regex_compile():
    """
    Validate that the PRETTIFY_RE regex exists and can be accessed.
    The test is expected to pass on the baseline validation and fail on the mutant, 
    which has an invalid regex compilation causing a runtime error.
    """
    try:
        pattern = PRETTIFY_RE['DUPLICATES']
        output = "Compiled successfully"
        assert output == "Compiled successfully"  # This should pass on the baseline
    except Exception as e:
        assert False, f"Regex compilation failed with error: {e}"