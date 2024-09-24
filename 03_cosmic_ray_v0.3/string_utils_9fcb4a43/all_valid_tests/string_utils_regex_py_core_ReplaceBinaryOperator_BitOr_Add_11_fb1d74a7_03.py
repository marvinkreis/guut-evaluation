from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    """Testing PRETTIFY_RE's regex against input patterns that may exploit the mutant change."""
    
    # Create input that strongly targets whitespace and duplication conditions
    test_input = """
    This is a test -- with -- some -- irregular spacing -- signs -- -- disturbed.
    
    -- Another -- edge case -- with a lot  --  of -- duplicates -- here -- like -- this --.
    
    Let's -- see -- how many -- duplicates -- we catch -- --- -- after -- change!
    """

    # Get matches using PRETTIFY_RE from correct code
    correct_matches = PRETTIFY_RE['DUPLICATES'].findall(test_input)

    # Expecting that the correct implementation recognizes a considerable number of duplicates, at least more than 5
    assert len(correct_matches) > 5, "The correct implementation should find multiple overlapping duplicate signs."

# This test will verify the behavior and assess if any match counting is affected by the mutant.