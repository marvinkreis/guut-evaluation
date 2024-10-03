from string_utils._regex import PRETTIFY_RE

def test_prettify_re_patterns():
    """
    Test each regex pattern in PRETTIFY_RE against a string with multiple spaces and punctuation.
    The mutant is expected to throw an OverflowError and not compile correctly,
    while the baseline should execute without errors.
    """
    test_string = "Hey....  this  is awful!! There is no doubt.  \n  Again!"
    
    for key, pattern in PRETTIFY_RE.items():
        matches = pattern.findall(test_string)
        print(f"Pattern: {key}, Matches: {matches}")
    
    # If we reach this point without exceptions, the baseline is passing.
    assert True