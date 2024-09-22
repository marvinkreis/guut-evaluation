from string_utils._regex import PRETTIFY_RE

def test_PRETTIFY_RE():
    # Define a multiline string where duplicate spaces should be caught
    test_string = """This  is a    test string.  
                    It has multiple lines.
                    Here  is yet    another    line."""

    # Attempt to find duplicate spaces in the original behavior
    matches_correct = PRETTIFY_RE['DUPLICATES'].findall(test_string)

    # Under correct regex behavior, we expect at least two matches for duplicates
    assert len(matches_correct) >= 2, "Expected at least two matches for duplicate spaces in the original regex behavior."

    # Now we can alter our test string slightly to check mutant behavior
    altered_test_string = """This is a test string.
                             It has multiple lines.
                             Here is yet another line."""
    
    # Simulating the mutant behavior: Remove 're.DOTALL' from conditions
    matches_mutant = PRETTIFY_RE['DUPLICATES'].findall(altered_test_string)

    # With the mutated regex, we expect that it fails to match as many
    assert len(matches_mutant) < len(matches_correct), "The mutant regex should detect fewer duplicates than the original."
