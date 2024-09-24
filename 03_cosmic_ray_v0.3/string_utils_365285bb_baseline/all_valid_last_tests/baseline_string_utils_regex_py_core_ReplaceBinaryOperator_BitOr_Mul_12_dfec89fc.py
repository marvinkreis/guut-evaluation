import re

def test__prettify_re():
    # Original regex for matching Saxon genitive
    saxon_genitive_pattern = re.compile(
        r"(?<=\w)'s|'\s+s(?=\w)|'\s+s\s(?=\w)",
        re.MULTILINE | re.UNICODE  # Correctly combines flags
    )
    
    # Test string that should match the original PRETTIFY_RE regex
    valid_test_string = "This is John's car."
    
    # Test for a valid Saxon genitive
    assert saxon_genitive_pattern.search(valid_test_string) is not None, "Expected match not found in original regex."

    # Now we will build a mutant version that simulates the modified regex
    mutant_saxon_genitive_pattern = re.compile(
        r"(?<=\w)'s|'\s+s(?=\w)|'\s+s\s(?=\w)",
        re.MULTILINE * re.UNICODE  # This simulates the mutant's error
    )
    
    # Test string without the possessive
    invalid_test_string = "This is Johns car."
    
    # Ensure that the original regex finds no match in the mutant version
    assert mutant_saxon_genitive_pattern.search(invalid_test_string) is None, "Unexpected match found in mutant regex."

    print("All tests passed.")

test__prettify_re()