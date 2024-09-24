from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    # Test input string that includes a percentage and should not match any spurious spaces around it
    test_string = "The value is 100% accurate."
    
    # This should be matched correctly by the original regex
    match = PRETTIFY_RE['DUPLICATES'].search(test_string)
    assert match is None, "Expected no duplicates to be found"

    # Now let's check a case that would be improperly matched by the mutant code. 
    # The mutant will improperly return a result where the original code does not,
    # due to the use of `//` instead of `|` in the regex.
    mutant_test_string = "Redundant   spaces between words."
    matches = PRETTIFY_RE['DUPLICATES'].findall(mutant_test_string)
    assert len(matches) > 0, "Expected to find duplicates due to the mutant's change."