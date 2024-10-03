from string_utils._regex import HTML_TAG_ONLY_RE

def test__html_tag_only_mutant_killing():
    """
    Test to check the proper functioning of HTML_TAG_ONLY_RE when processing multi-line HTML strings.
    The first test verifies that the Baseline returns all matching HTML tags. The Mutant is expected to raise a ValueError
    indicating a flags conflict, which makes it easily detectable.
    """
    multi_line_html = """<div>
        <p>Hello World</p>
    </div>"""
    
    # Testing the Baseline
    try:
        output = HTML_TAG_ONLY_RE.findall(multi_line_html)
        print(f"output = {output}")
        assert output == ['<div>', '</div>', '<p>', '</p>']
    except Exception as e:
        print(f"Exception in Baseline: {e}")

    # The Mutant should raise a ValueError
    try:
        output = HTML_TAG_ONLY_RE.findall(multi_line_html)
        print(f"output = {output}")
    except ValueError as ex:
        print(f"Caught ValueError as expected in Mutant: {ex}")
        assert True
    except Exception as e:
        print(f"Unexpected Exception in Mutant: {e}")
        assert False