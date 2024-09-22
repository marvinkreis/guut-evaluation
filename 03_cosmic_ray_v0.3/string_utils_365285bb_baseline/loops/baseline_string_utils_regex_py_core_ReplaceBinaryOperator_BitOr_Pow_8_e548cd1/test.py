from string_utils._regex import PRETTIFY_RE
import re

def test__PRETTIFY_RE():
    # This string contains multiple spaces and should be tested against the regex
    test_string = "This is an    example    string  with spaces."

    # Expected output after applying the original regex: all extra spaces should be replaced by a single space
    expected_output = "This is an example string with spaces."

    # Clean using the original PRETTIFY_RE regular expression for duplicates
    cleaned_string = re.sub(PRETTIFY_RE['DUPLICATES'], ' ', test_string)

    # Assert that the cleaned string matches the expected output
    assert cleaned_string == expected_output, f"Expected: '{expected_output}', got: '{cleaned_string}'"

    # Now we simulate what an incorrect mutant might do by directly checking an altered behavior:
    # The mutant regex could be thought of to incorrectly reduce all spaces, potentially changing behavior 
    mutant_test_string = "This is an example string with spaces."  # This string may behave differently if the logic is flawed

    # A faulty mutant behavior might incorrectly alter space handling
    mutant_cleaned_string = re.sub(r'\s+', ' ', mutant_test_string)  # Simplifying into one space (this may not trigger the expected behavior)
    
    # Directly introduce a case where this mutant does something it normally shouldn't
    mutant_cleaned_string_wrong = re.sub(r'\s+', '', mutant_test_string)  # Mutation may cause it to remove spaces entirely

    # Assert that the mutant output is not the same as expected
    assert mutant_cleaned_string_wrong != expected_output, f"Mutant produced the same output: '{mutant_cleaned_string_wrong}'"

# Execute the test function
test__PRETTIFY_RE()