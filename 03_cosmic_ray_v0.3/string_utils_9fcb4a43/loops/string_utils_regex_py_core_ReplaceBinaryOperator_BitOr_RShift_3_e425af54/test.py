from string_utils._regex import HTML_RE

def test__HTML_RE():
    """Test HTML_RE to ensure it behaves correctly with various HTML inputs, particularly edge cases."""
    
    # Define a set of HTML samples with both valid and invalid structures
    html_samples = [
        "<html><head></head><body><h1>Hello World</h1></body></html>",  # Valid HTML
        "<html><head>\n<title>Title</title>\n</head><body><h1>Hello World</h1></body></html>",  # Valid with newlines
        "<html><head><style>body { margin: 0; }</style></head><body><h1>Hello World</h1></body></html>",  # Style tag
        "<html><head></head><body><script>console.log('Hello');</script><h1>Hello World</h1></body></html>",  # Script tag
        "<html><head></head><body><div><span>Hello</span></div></body></html>",  # Nested tags
        "<!DOCTYPE html><html><head></head><body><h1>Data</h1><!-- comment --></body></html>",  # Doctype with comment
        "<html><head></head><body><h1></h1></body></html>",  # Empty tag
        "<html><head></head><body><h1>&nbsp;</h1></body></html>",  # Non-breaking space in h1
        "<html><head><body><h1>Unclosed Tag</h1></html>",  # Malformed HTML with unclosed tags
        "<><html><></html>",  # Invalid/malformed HTML
        "!!notanhtml!!",  # Completely invalid
        "<html><head><title></title></head><body>Malformed</body>!!!<title></title></html>",  # Invalid structure
        "<html><head></head><body><h1>Hello World!!<h1></body></html>",  # Closing tag problems
    ]

    for sample in html_samples:
        result = HTML_RE.match(sample)
        if sample in ["<><html><></html>", "!!notanhtml!!"]:
            assert result is None, f"HTML_RE incorrectly matched on invalid input: {sample}"
        else:
            assert result is not None, f"HTML_RE failed to match on valid input: {sample}"

# Run the test
test__HTML_RE()