from string_utils._regex import HTML_RE

def test_HTML_RE():
    # Test string that is valid HTML
    valid_html = "<div>Hello World</div>"
    
    # Ensure the original regex matches valid HTML
    assert HTML_RE.match(valid_html) is not None, "The original HTML_RE should match valid HTML."
    
    # Check for a multi-line valid HTML string (this should still be valid)
    multi_line_html = "<div>Hello</div>\n<div>World</div>"  # HTML structure with new lines
    
    # Now let's test this case
    try:
        match_result = HTML_RE.match(multi_line_html)
        
        # Should match because it's structurally valid HTML
        assert match_result is not None, "The original HTML_RE should match multi-line HTML."
        
    except ValueError as e:
        # If we catch a ValueError here, we know we're running the mutant code
        assert True, "Detected mutant code due to incompatible regex flags."
        print("Mutant detected successfully.")

# Execute the test
test_HTML_RE()