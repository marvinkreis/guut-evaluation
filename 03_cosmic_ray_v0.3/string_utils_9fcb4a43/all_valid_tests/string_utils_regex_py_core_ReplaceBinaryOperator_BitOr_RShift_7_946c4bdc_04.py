def test__mutant_detection():
    """Directly test if PRETTIFY_RE can still perform regex operations; should fail with mutant."""
    from string_utils._regex import PRETTIFY_RE

    sample_text = "The sun is shining brightly on the bright sun!"

    try:
        # Check matching on keys that should produce identifiable behavior
        match_duplicates = PRETTIFY_RE['DUPLICATES'].search(sample_text)
        if match_duplicates:
            print("DUPLICATES regex found a match:", match_duplicates.group())
        else:
            print("DUPLICATES regex did not find a match.")

        match_upper = PRETTIFY_RE['UPPERCASE_FIRST_LETTER'].search(sample_text)
        if match_upper:
            print("UPPERCASE_FIRST_LETTER regex found a match:", match_upper.group())
        else:
            print("UPPERCASE_FIRST_LETTER regex did not find a match.")

    except Exception as e:
        print(f"An error occurred during regex processing: {e}")

# Run the test
test__mutant_detection()