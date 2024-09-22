from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only():
    # Prepare an HTML string with multiple tags
    html_string = "<div><p>This is a paragraph.</p><!-- Comment --></div>"

    # Use findall to capture all matches of HTML tags
    matches = HTML_TAG_ONLY_RE.findall(html_string)

    # Assert that we should have some matches found
    assert len(matches) > 0, "Expected to find at least one HTML tag match."
    
    # Print captured matches for debugging
    print("Matches found:", matches)

    # Flatten the matches since they are tuples and just check the first element
    flattened_matches = [match[0] for match in matches]

    # Check if all captured matches resemble valid HTML tags or comments
    assert all(tag.startswith("<") and tag.endswith(">") for tag in flattened_matches), \
        "All matches should resemble valid HTML tags or comments."

# Run the test
test__html_tag_only()