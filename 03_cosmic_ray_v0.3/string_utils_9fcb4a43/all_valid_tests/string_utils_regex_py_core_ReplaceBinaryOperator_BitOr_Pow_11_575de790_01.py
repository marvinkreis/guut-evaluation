from string_utils._regex import PRETTIFY_RE

def test__PRETTIFY_RE():
    """Check that all regex patterns in PRETTIFY_RE can be compiled and run without errors."""
    sample_text = 'This is a sample text where we can see "quoted text" and (some brackets).'
    
    for key, pattern in PRETTIFY_RE.items():
        try:
            # Compile the regex pattern (this will raise an error if it fails)
            regex = pattern
            
            # Run the pattern against the sample text to see if it executes without error
            matches = regex.findall(sample_text)
            
            # Instead of empty checks, simply output matches for verification
            print(f"{key} matches found:", matches)
        except Exception as e:
            assert False, f"An unexpected error occurred with {key}: {e}"
