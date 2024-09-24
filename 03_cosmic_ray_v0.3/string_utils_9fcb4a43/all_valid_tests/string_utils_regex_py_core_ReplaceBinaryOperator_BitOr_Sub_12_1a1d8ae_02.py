import re

def test__saxon_genitive_regex():
    """Tests the SAXON_GENITIVE regex against a variety of inputs to confirm correct matching."
    """
    test_strings = [
        "Tom's",  # Should match
        "Mary's book",  # Should also match
        "Hello world",  # Should not match
        "The dog is Alice's pet.",  # Should match
    ]

    # Correct implementation
    try:
        correct_pattern = re.compile(r"(?<=\w)'s", re.UNICODE)

        for test_string in test_strings:
            correct_match = correct_pattern.search(test_string)
            assert (test_match := correct_match) is not None or "The correct regex should find a match for '{}'.".format(test_string)

    except ValueError:
        assert False, "Correct code should compile without error."

    # Attempting to compile the mutant version should fail
    try:
        mutant_pattern = re.compile(r"(?<=\w)'s", re.MULTILINE - re.UNICODE)
        assert False, "Mutant should raise a ValueError for incompatible flags."
    except ValueError:
        pass  # This confirms that the mutant behaves as expected by raising an error.