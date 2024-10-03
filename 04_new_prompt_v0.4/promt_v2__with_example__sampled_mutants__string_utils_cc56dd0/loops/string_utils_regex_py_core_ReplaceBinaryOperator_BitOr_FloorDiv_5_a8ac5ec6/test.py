from string_utils._regex import HTML_TAG_ONLY_RE

def test_html_tag_only_re_mutant_killing_with_case_sensitive():
    """
    Test the HTML_TAG_ONLY_RE regex with mixed-case HTML input.
    The baseline should match all tags, while the mutant will fail to match
    due to the absence of the re.IGNORECASE flag.
    """
    mixed_case_html_input = "<Html><BODY></body></Html>"
    output = HTML_TAG_ONLY_RE.findall(mixed_case_html_input)
    print(f"Output: {output}")
    assert len(output) == 4, f"Expected matches for all tags, but got {len(output)}: {output}"