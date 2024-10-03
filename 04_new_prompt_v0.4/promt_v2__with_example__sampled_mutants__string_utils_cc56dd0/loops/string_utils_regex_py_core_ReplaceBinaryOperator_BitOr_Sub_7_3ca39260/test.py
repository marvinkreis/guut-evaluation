from string_utils._regex import PRETTIFY_RE  # This pulls the regex for testing both mutant and baseline executions.

def test_import_prettyfy_regex():
    """
    Test the importation and behavior of PRETTIFY_RE regex patterns. 
    The baseline should import successfully, while the mutant should raise a ValueError
    indicating incompatible regex flags due to the alteration.
    """
    try:
        # Accessing a specific regex pattern
        _ = PRETTIFY_RE['DUPLICATES']
        print("Output: Import successful.")
    except ValueError as ve:
        # Expect this to be raised in the mutant
        assert str(ve) == "ASCII and UNICODE flags are incompatible", f"Unexpected ValueError: {ve}"
    except Exception as e:
        # This would catch other types of exceptions
        assert False, f"Unexpected exception raised: {e}"