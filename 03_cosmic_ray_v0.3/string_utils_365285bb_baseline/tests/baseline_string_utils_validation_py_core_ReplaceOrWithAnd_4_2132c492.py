from string_utils.validation import is_email

def test_is_email_mutant_detection():
    # Test case with an email that has two consecutive dots in the username
    mutant_email = 'my..email@gmail.com'  # This should return False

    # Assert that the original function returns False (valid behavior)
    assert is_email(mutant_email) == False, "Original function should return False for the mutant email."

    # To conceptually determine the mutant's behavior, validate that the mutant would incorrectly return True,
    # but we cannot directly invoke the mutant version; this comment serves as an indication of expected behavior if the mutant was active.

# Running the test function
test_is_email_mutant_detection()