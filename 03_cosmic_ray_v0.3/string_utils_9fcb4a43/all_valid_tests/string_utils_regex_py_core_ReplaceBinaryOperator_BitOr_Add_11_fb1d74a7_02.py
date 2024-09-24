from string_utils._regex import PRETTIFY_RE

def test__prettify_re():
    """Testing PRETTIFY_RE's duplication detection against input that could result in different matches due to regex change."""
    
    # Input string that aims to exploit the logic behind the regex change
    test_input = """
    Test -- 
    Here is a sample  --  
    Here -- 
    Another --   case with lots --   of spaces. 
    --   end.
    """

    # Check matches against PRETTIFY_RE
    correct_matches = PRETTIFY_RE['DUPLICATES'].findall(test_input)
    
    # Expecting certain patterns relating to whitespace and repeated signs
    # If duplicates are detected, this should be greater than 3 based on the input structure
    assert len(correct_matches) > 3, "The correct implementation should recognize multiple duplicate signs."

# When executed, this test should:
# - Pass with the correct implementation by observing the expected number of duplicates.
# - Fail with the mutant version where the regex may perform incorrectly due to the logical change made in the mutation.