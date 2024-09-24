from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # Sample text with deliberate duplicate spaces
    sample_text = """This is a sample text with multiple    spaces.   
    
    And     this line      has   multiple   spaces. 
    """
    
    # Check for duplicates in the original regex
    original_matches = PRETTIFY_RE['DUPLICATES'].findall(sample_text)
    assert len(original_matches) > 0, "Original regex should find duplicate spaces."

    # Multiline text to ensure spaces across lines are detected
    multiline_text = """This line has excessive    spaces   
    and    also multiple      lines     with   spaces. 
    """
    
    multiline_matches = PRETTIFY_RE['DUPLICATES'].findall(multiline_text)
    assert len(multiline_matches) > 0, "Should find duplicates across newlines."

    # Clean text without duplicates
    clean_text = """This text should not have any duplicates or extra spaces.
This is a clean string without extra spacing.
"""

    clean_matches = PRETTIFY_RE['DUPLICATES'].findall(clean_text)
    assert len(clean_matches) == 0, "Should detect no duplicates."

    # Text with specific duplicate structures to expose weaknesses in regex for mutants
    forced_repetition_text = """This line has    
    excessive spaces   leading to duplicate
    issues  that should     be captured.
    """

    forced_matches = PRETTIFY_RE['DUPLICATES'].findall(forced_repetition_text)
    assert len(forced_matches) > 3, "Original regex should detect multiple duplicates found."

    print("Test passed!")

# Run the test
test_PRETTIFY_RE()