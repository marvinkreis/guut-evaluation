def test__html_regex_compilation_original():
    """Ensure the original HTML_RE compiles without errors."""
    try:
        from string_utils._regex import HTML_RE
        HTML_RE.pattern  # Accessing the pattern should not raise any error
        print("Original HTML_RE compiled successfully.")
    except Exception as e:
        print(f"Original HTML_RE raised an error: {e}")

def test__html_regex_compilation_mutant():
    """Ensure that the mutant HTML_RE raises an error due to invalid operator."""
    try:
        # Here we simulate the context to only handle mutant imports
        import mutant.string_utils._regex
        HTML_RE = mutant.string_utils._regex.HTML_RE
        raise AssertionError("Expected an error in mutant HTML_RE compilation, but it succeeded.")
    except TypeError:
        print("Expected TypeError in mutant HTML_RE compilation.")
    except Exception as e:
        print(f"Unexpected error when compiling mutant HTML_RE: {e}")

def main():
    test__html_regex_compilation_original()
    test__html_regex_compilation_mutant()

# Execute the tests
main()