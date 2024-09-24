from string_utils._regex import HTML_TAG_ONLY_RE

def test_HTML_TAG_ONLY_RE():
    # Test string with various HTML tags and comments
    test_string = "<html><body><h1>Title</h1><!-- Comment --></body></html>"
    
    # Actual matches based on what we see from the regex
    matches = HTML_TAG_ONLY_RE.findall(test_string)

    # Print the matches for debugging purposes
    print("Matches:", matches)
    
    extracted_matches = []

    for match in matches:
        for group in match:
            if group:  # Append non-empty groups only
                extracted_matches.append(group)

    # Here we build our expected output based on the regex output
    expected_matches = [
        "<html>",
        "<body>",
        "<h1>",
        "<!-- Comment -->",
        "</body>",
        "</html>"
    ]

    # The assertion should compare the actual output from extraction with what we expect
    assert extracted_matches == expected_matches, f"Expected {expected_matches}, but got {extracted_matches}"

# Run the test
test_HTML_TAG_ONLY_RE()