from string_utils._regex import HTML_RE

def test__html_regex_with_conflicting_markup():
    """Test HTML_RE against complex markup that could expose mutant behavior through regex pitfalls."""

    # Intentionally malformed and complex HTML-like syntax
    conflicting_markup = "<div><!-- <p>Test <div>Multiple<!-- nested --> tags <p></div><!-- </p> -->"

    # Testing with the correct implementation
    match_builtin = HTML_RE.match(conflicting_markup)
    if match_builtin:
        print("Matched groups in correct implementation:", match_builtin.groups())
    else:
        print("Correct implementation did not match as expected.")

    # Testing with the mutant implementation
    mutant_match = HTML_RE.match(conflicting_markup)
    if mutant_match:
        print("Matched groups in mutant implementation:", mutant_match.groups())
    else:
        print("Mutant implementation did not match as expected.")