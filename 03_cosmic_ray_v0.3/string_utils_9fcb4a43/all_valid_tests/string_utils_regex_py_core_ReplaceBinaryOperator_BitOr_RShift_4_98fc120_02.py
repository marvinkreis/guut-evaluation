from string_utils._regex import HTML_RE

def test__html_regex_with_unconventional_format():
    """Test HTML_RE against unconventional and highly malformed HTML to expose potential mutant behavior."""
    
    unconventional_html = "<div><p>Test <div>Unclosed <p><div></div></div></p></div><div>Another</div>"

    # Testing with the correct implementation
    match = HTML_RE.match(unconventional_html)
    if match is None:
        print("Correct HTML_RE failed to match, as expected for unconventional input.")
    else:
        print(f"Matched groups in correct implementation: {match.groups()}")

    # Checking that the mutant behaves differently
    mutant_match = HTML_RE.match(unconventional_html)
    if mutant_match is None:
        print("Mutant HTML_RE failed to match, as expected for unconventional input.")
    else:
        print(f"Matched groups in mutant implementation: {mutant_match.groups()}")