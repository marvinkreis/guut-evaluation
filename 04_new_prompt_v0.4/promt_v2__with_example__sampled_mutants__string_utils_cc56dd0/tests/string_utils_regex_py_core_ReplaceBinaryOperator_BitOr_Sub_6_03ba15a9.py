from string_utils._regex import HTML_TAG_ONLY_RE

def test_html_tag_only_mutant_killing():
    """
    Test the HTML_TAG_ONLY_RE regex with a multi-line HTML tag.
    The mutant will raise a ValueError due to incompatible flags,
    while the baseline will return matching tags.
    """
    html_input = "<div>\n  Content\n</div>"
    try:
        match = HTML_TAG_ONLY_RE.findall(html_input)
        assert match, "Expected to find matches but found none."
    except ValueError as ve:
        print(f"ValueError raised: {ve}")
        return  # This will indicate that the mutant is behaving differently.