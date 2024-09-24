try:
    from string_utils._regex import PRETTIFY_RE

    def test__prettify_re():
        # Step 1: Ensure necessary keys exist in PRETTIFY_RE
        assert 'DUPLICATES' in PRETTIFY_RE, "DUPLICATES key should exist in PRETTIFY_RE."
        assert 'RIGHT_SPACE' in PRETTIFY_RE, "RIGHT_SPACE key should exist in PRETTIFY_RE."

        # Step 2: Validate the 'DUPLICATES' regex
        duplicates_regex = PRETTIFY_RE['DUPLICATES']
        assert duplicates_regex.pattern is not None, "DUPLICATES regex should have a valid pattern."

        # Test matching multiple spaces
        assert duplicates_regex.search("This is    a test.") is not None, "Expected a match for multiple spaces."
        assert duplicates_regex.search("This is a test.") is None, "Expected no match for single spaces."

        # Step 3: Validate the 'RIGHT_SPACE' regex
        right_space_regex = PRETTIFY_RE['RIGHT_SPACE']
        assert right_space_regex.pattern is not None, "RIGHT_SPACE regex should have a valid pattern."

        # Test matching space before a comma
        assert right_space_regex.search("This , is a test.") is not None, "Expected a match for space before a comma."
        assert right_space_regex.search("This is a test, correctly.") is None, "Expected no match for correct spacing."
        
    # Execute the test
    test__prettify_re()

except Exception as e:
    # Upon exception, print the error to diagnose mutant detection
    print(f"Test failed with error: {str(e)}")