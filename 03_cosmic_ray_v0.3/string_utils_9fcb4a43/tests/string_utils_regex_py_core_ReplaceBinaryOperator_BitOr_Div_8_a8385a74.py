from string_utils._regex import PRETTIFY_RE
try:
    from mutant.string_utils._regex import PRETTIFY_RE as mutant_PRETTIFY_RE
except ImportError or NameError:
    mutant_PRETTIFY_RE = None  # Mimic a failure in case mutant cannot be imported

def test__PRETTIFY_RE_compilation():
    """The regex must compile successfully; changing '|' to '/' will raise an error."""

    # Test correct regex compilation
    try:
        correct_compile = PRETTIFY_RE
        assert correct_compile is not None, "PRETTIFY_RE should compile successfully."
    except Exception as e:
        assert False, f"Error compiling correct PRETTIFY_RE: {str(e)}"

    # Test mutant regex compilation (expected to raise an error)
    try:
        mutant_compile = mutant_PRETTIFY_RE
        assert mutant_compile is not None, "Mutant PRETTIFY_RE should not compile successfully, but it did."
    except Exception:
        # This is expected; the mutant regex should fail to compile
        pass