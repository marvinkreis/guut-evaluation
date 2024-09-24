# Let's define a function to directly inspect the 'DUPLICATES' regex to ensure it works as thought.
from string_utils._regex import PRETTIFY_RE

def inspect_duplicates_regex():
    """ Inspect the DUPLICATES regex pattern. """
    duplicates_regex = PRETTIFY_RE['DUPLICATES']
    print(f"DUPLICATES regex pattern: {duplicates_regex.pattern}")

    # Define a list of test strings for direct inspection
    test_strings = [
        "This is a test.",
        "A    B",
        "Match: (example)",
        "Quote: \"hello world\"",
        "Error: (   )",
        "Multiple spaces  , here",  # additional input for thorough testing
    ]
    
    print("\nTesting DUPLICATES regex:")
    for s in test_strings:
        match_result = duplicates_regex.search(s) is not None
        print(f"Input: '{s}' => Match: {match_result}")

inspect_duplicates_regex()