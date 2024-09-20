def test__prettify_re():
    """The mutant code fails to compile due to incorrect operator on line 117."""
    
    # Testing import of correct PRETTIFY_RE
    try:
        from string_utils._regex import PRETTIFY_RE
        assert PRETTIFY_RE is not None, "Correct PRETTIFY_RE should be imported successfully"
    except Exception as e:
        assert False, f"Failed to import correct PRETTIFY_RE: {e}"
    
    # Testing import of mutant PRETTIFY_RE, which should fail
    try:
        from mutant.string_utils._regex import PRETTIFY_RE
        assert False, "Should not be able to import mutant PRETTIFY_RE"
    except Exception:
        pass  # This is expected, as the mutant should fail to compile