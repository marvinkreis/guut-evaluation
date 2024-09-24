import re

# Original regex that we expect to work correctly
def original_uppercase_after_sign(text):
    regex = re.compile(r'([.?!])\s([A-Z])')
    return [match[1] for match in regex.findall(text)]

# Mutant version of the regex which should fail or behave differently
def mutant_uppercase_after_sign(text):
    # This mutation should generate incorrect behavior, e.g., using wrong operator(s)
    regex = re.compile(r'([.?!])\s\w', re.MULTILINE | re.UNICODE)  # Intentional error
    return [match[1] for match in regex.findall(text)]

def test_prettify_uppercase_after_sign():
    # Test input string
    test_string = "Hello world! This Is A Test. Are you okay?"

    # Test original implementation
    expected_matches = ['T', 'A']
    found_uppercases = original_uppercase_after_sign(test_string)

    print("Original Matches:", found_uppercases)

    # Validate the original regex output
    assert sorted(found_uppercases) == sorted(expected_matches), (
        f"Expected {expected_matches}, but got {found_uppercases}."
    )

    # Test mutant implementation
    try:
        mutant_uppercases = mutant_uppercase_after_sign(test_string)
        print("Mutant Matches:", mutant_uppercases)

        # This should throw an error or yield incorrect results
        assert False, "Mutant code did not fail as expected."
    except Exception as e:
        print("Mutant code raised an exception as expected:", str(e))
        # Confirming that the mutant behavior is recognized

# Run the test function
test_prettify_uppercase_after_sign()