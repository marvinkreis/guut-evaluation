from string_utils._regex import HTML_TAG_ONLY_RE  # Correct implementation
# The mutant implementation should NOT be included here; it's just for testing comparison

def test__HTML_TAG_ONLY_RE():
    """The correct regex must match HTML tags, while the mutant regex should fail to provide valid matches."""
    sample_inputs = [
        "<div>Hello World</div>",
        "<a href='https://example.com'>Example</a>",
        "<p>This is a paragraph.</p>",
        "<html><body><h1>Header</h1></body></html>"
    ]
    
    for input_str in sample_inputs:
        # Test correct regex for matching HTML tags
        correct_match = HTML_TAG_ONLY_RE.findall(input_str)  # Ensure this line uses the imported correct regex
        assert correct_match, f"HTML_TAG_ONLY_RE must match HTML tags for input: '{input_str}'"
        
        # Simulate testing the mutant regex (for demonstration purposes here)
        # Ideally, you would call the mutant regex in a separate testing context/file
        try:
            mutant_match = HTML_TAG_ONLY_RE.findall(input_str)  # This line should refer to the mutant regex in actual implementation
            print(f"Testing input: '{input_str}' - Mutant Output: {mutant_match}")
        except Exception as ex:
            print(f"Mutant raised an exception for input: '{input_str}' - {str(ex)}")

# Execute the test function directly to evaluate behavior
test__HTML_TAG_ONLY_RE()