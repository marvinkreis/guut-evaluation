import re

# Improved regex to handle HTML tags and comments properly
HTML_TAG_ONLY_RE = re.compile(
    r'^\s*(<!--.*?-->\s*)*<([a-z][a-z0-9]*)\b[^>]*(?<!/)>[^<]*</\2>\s*$|^<([a-z][a-z0-9]*)\s*/>$',
    re.IGNORECASE | re.DOTALL
)

def test_HTML_TAG_ONLY_RE():
    # Valid HTML cases
    assert HTML_TAG_ONLY_RE.match("<div>Hello World</div>") is not None, \
        "Failed to match valid HTML: <div>Hello World</div>"
    
    assert HTML_TAG_ONLY_RE.match("<br />") is not None, \
        "Failed to match valid self-closing HTML tag: <br />"

    # Invalid HTML cases
    # 1. Invalid tag without closing
    assert HTML_TAG_ONLY_RE.match("<div>") is None, \
        "Matched an invalid HTML tag without a closing tag: <div>"

    # 2. Just plain text
    assert HTML_TAG_ONLY_RE.match("This is plain text.") is None, \
        "Matched plain text as valid HTML."

    # 3. Malformed self-closing tag (missing '/')
    assert HTML_TAG_ONLY_RE.match("<br>") is None, \
        "Matched an invalid self-closing HTML tag without a slash: <br>"

    # 4. Nested improperly
    assert HTML_TAG_ONLY_RE.match("<div><span></div></span>") is None, \
        "Matched improperly nested HTML tags: <div><span></div></span>"

    # 5. Malformed comment structure missing closing
    assert HTML_TAG_ONLY_RE.match("<!-- This is a comment") is None, \
        "Matched a comment without a closing: <!-- This is a comment"

    # 6. Invalid broken structure
    assert HTML_TAG_ONLY_RE.match("<div<><p>Text</p>") is None, \
        "Matched an invalid broken HTML structure: <div<><p>Text</p>"

    # 7. Valid with comment
    assert HTML_TAG_ONLY_RE.match("<!-- Valid comment --> <p>Text</p>") is not None, \
        "Failed to match valid HTML with comments: <!-- Valid comment --> <p>Text</p>"

# Run the modified test function
test_HTML_TAG_ONLY_RE()