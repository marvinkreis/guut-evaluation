import re

# The original regex for detecting Saxon genitive
SAXON_GENITIVE_RE = re.compile(
    r'(?<=\w)\'s|'              # Matches: "word's"
    r'(?<=\w)\s\'s\s',           # Matches: " word's "
    re.MULTILINE | re.UNICODE
)

def test_saxon_genitive_detection():
    # Define test cases with expected outcomes
    test_cases = {
        "John's book": True,       # Should match
        "the dog's owner": True,   # Should match
        "Chris is running": False,  # Should not match
        "The cat's toy is here": True,  # Should match
        "without an apostrophe": False,  # Should not match
        "Sarah's cat's toy": True,  # Should match
        "The books aren't mine": False  # Should not match
    }

    matched_count = 0
    
    for case, expected in test_cases.items():
        match_found = bool(SAXON_GENITIVE_RE.search(case))
        assert match_found == expected, f"Original Text: '{case}' - Expected {expected}, Found {match_found}"
        
        if match_found:
            matched_count += 1

    assert matched_count == 4, f'Expected total matches to be 4, but found {matched_count}.'

# Define mutant logic where the faulty regex won't match
def mutant_saxon_genitive_detection():
    # Mutant regex pattern (using / will break the regex)
    try:
        MUTANT_SAXON_GENITIVE_RE = re.compile(
            r'(?<=\w)\'s/'              # This will lead to broken functionality
            r'(?<=\w)\s\'s\s'
        )

        # Test the same cases against mutant logic
        mutant_cases = [
            "John's book",      # Should match in original
            "the dog's owner",  # Should match in original
            "Chris is running", # Should not match in original
            "The cat's toy is here",  # Should match in original
            "without an apostrophe",  # Should not match in original
            "Sarah's cat's toy",  # Should match in original
            "The books aren't mine"  # Should not match in original
        ]

        for case in mutant_cases:
            match_found = bool(MUTANT_SAXON_GENITIVE_RE.search(case))
            # Since the regex should not match valid patterns, assert as below
            assert not match_found, f"Mutant Detected! Text: '{case}' should NOT match, Found: {match_found}."

    except re.error as e:
        print(f"Mutant regex caused a compilation error: {e}")

# Execute tests
if __name__ == "__main__":
    test_saxon_genitive_detection()  # Valid patterns
    mutant_saxon_genitive_detection() # Validates failure to match in mutant
    print("All tests completed!")