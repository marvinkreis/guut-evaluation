from string_utils._regex import PRETTIFY_RE

def test__saxon_genitive():
    """
    Test the SAXON_GENITIVE regex against multiple inputs to check for proper matching of possessives.
    The baseline version should return no matches as expected, while the mutant fails with an OverflowError.
    """
    test_strings = [
        "John's book",       # simple possessive
        "This is John's.",    # possessive with punctuation
        "Does Jane's cat exist?",  # possessive with punctuation and question
        "Alice's dog\nis cute.",  # possessive with line break
        "There is no possessive here.",  # no possessive
    ]
    
    for s in test_strings:
        match = PRETTIFY_RE['SAXON_GENITIVE'].findall(s)
        print(f"Testing '{s}': found matches: {match}")