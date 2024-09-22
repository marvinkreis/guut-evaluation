try:
    from string_utils._regex import HTML_RE

    # A valid HTML string
    valid_html = "<html><body><h1>Title</h1></body></html>"
    # An invalid HTML string, example with unclosed tags
    invalid_html = "<html><body><h1>Title</h1><body></html>"

    # 1. Test to ensure valid HTML string matches
    valid_match = HTML_RE.match(valid_html)
    if valid_match is None:
        raise AssertionError("The HTML_RE should match valid HTML")

    # 2. Test to ensure invalid HTML string does not match
    invalid_match = HTML_RE.match(invalid_html)
    if invalid_match is not None:
        raise AssertionError("The HTML_RE should not match invalid HTML")

    # 3. Confirm that an empty string does not match
    if HTML_RE.match("") is not None:
        raise AssertionError("The HTML_RE should not match an empty string")

    # 4. Confirm a plain non-HTML string does NOT match
    if HTML_RE.match("This is just a plain string.") is not None:
        raise AssertionError("The HTML_RE should not match a non-HTML string")

except ImportError as e:
    print("Import failed:", e)
except AssertionError as e:
    print("A test failed for HTML match validation:", e)
except overflowError:
    print("An error related to integer overflow occurred:", e)
except Exception as e:
    print("An unexpected error occurred:", e)