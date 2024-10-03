from string_utils._regex import PRETTIFY_RE

def test__uppercase_after_sign_with_non_ascii():
    """
    This tests whether the UPPERCASE_AFTER_SIGN regex properly identifies uppercase letters that follow punctuation,
    specifically in cases where non-ASCII characters are present. The input 'Hello! éA' should raise a ValueError 
    in the mutant due to incompatible regex flags.
    """
    test_string = "Hello! éA"
    
    # The baseline should produce a match without raising any exception
    output = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string)
    assert output == ['! é']
    
    # This will raise the ValueError in the mutant and won't raise it in the baseline
    output_mutant = PRETTIFY_RE['UPPERCASE_AFTER_SIGN'].findall(test_string)

test__uppercase_after_sign_with_non_ascii()