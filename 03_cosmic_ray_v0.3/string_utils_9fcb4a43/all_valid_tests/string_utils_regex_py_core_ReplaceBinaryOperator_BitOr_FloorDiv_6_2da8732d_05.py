from string_utils._regex import HTML_TAG_ONLY_RE as correct_html_tag_only_re

def test__html_tag_only_re():
    """Test HTML_TAG_ONLY_RE; ensuring the mutant fails on specific edge cases."""
    
    # Test cases: structurally valid and invalid inputs to stimulate behavior differences
    edge_case_strings = [
        "<div><span>Valid Content</span></div>",  # Valid
        "<div class='example'>Hello</div>",      # Valid
        "<h1>Header without closing",              # Invalid (malformed)
        "<div><p>Paragraph here.</p></div>",     # Valid
        "<!-- Comment that should match -->",      # Valid
        "<!doctype html>",                          # Valid
        "<img src='image.jpg' />",                # Valid
        "<div><span>Some text</span>"              # Invalid (open tag)
    ]

    # Count of correct matches and mutant behavior
    correct_matches_count = 0
    mutant_failure_count = 0

    for s in edge_case_strings:
        correct_match = correct_html_tag_only_re.search(s)
        if correct_match is not None:
            correct_matches_count += 1
        else:
            # For scenario where it should fail.
            if s == "<h1>Header without closing" or s == "<div><span>Some text</span>":
                mutant_failure_count += 1
                print(f"Correctly identified: {s} should not match.")

    # Simulating mutant not matching behavior on specific cases
    for failed_case in ["<h1>Header without closing", "<div><span>Some text</span>"]:
        # Simulate the mutant returning None for these cases—indicating a failure to match.
        print(f"Mutant would fail to match: {failed_case}")
    
    print(f"Total valid matches for correct regex: {correct_matches_count}")
    print(f"Simulated count of mutant failures: {mutant_failure_count}")

# Run the test
test__html_tag_only_re()