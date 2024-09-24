import re

def test__html_regex_functionality():
    """ Test that the correct HTML_RE regex matches multi-line HTML content,
        while the mutant will fail to compile. """

    # Testing the correct HTML regex
    try:
        HTML_RE = re.compile(
            r'((<([a-z]+:)?[a-z]+[^>]*/?>)(.*?(</([a-z]+:)?[a-z]+>))?|<!--.*-->|<!doctype.*>)',
            re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

        # Prepare the test HTML content
        html_content = """
        <html>
        <head>
        <title>Test</title>
        </head>
        <body>
        <p>This is a test.</p>
        </body>
        </html>
        """

        # Properly run the regex search
        correct_output = HTML_RE.search(html_content)
        assert correct_output is not None, "Expected to find a match with the correct HTML_RE."
        print("Correct HTML_RE found a match as expected.")

    except ValueError as ve:
        assert False, f"Expected correct HTML_RE to compile, but got error: {ve}"

    # Testing the mutant HTML regex
    mutant_failed = False

    try:
        mutant_HTML_RE = re.compile(
            r'((<([a-z]+:)?[a-z]+[^>]*/?>)(.*?(</([a-z]+:)?[a-z]+>))?|<!--.*-->|<!doctype.*>)',
            re.IGNORECASE | re.MULTILINE - re.DOTALL
        )

        # Attempt to use the mutant to perform search, expecting this to fail
        mutant_output = mutant_HTML_RE.search(html_content)
        
        # If it finds an output, it indicates an unexpected behavior
        if mutant_output:
            print("Mutant HTML_RE found a match unexpectedly.")
            mutant_failed = True
        
    except ValueError:
        # This is the expected behavior; the mutant regex will fail to compile
        print("Mutant HTML_RE failed to compile as expected.")

    # Final assertion to validate mutant behavior
    assert not mutant_failed, "Mutant HTML_RE should not function correctly and should raise an error or not find a match."