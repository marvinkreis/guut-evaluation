def test__PRETTIFY_RE():
    """The mutant change in PRETTIFY_RE should cause a compilation error."""
    try:
        from string_utils._regex import PRETTIFY_RE
        # If no exception is raised while importing, test should pass
        assert True, "PRETTIFY_RE compiled successfully"
    except Exception as e:
        assert False, f"PRETTIFY_RE failed to compile: {e}"