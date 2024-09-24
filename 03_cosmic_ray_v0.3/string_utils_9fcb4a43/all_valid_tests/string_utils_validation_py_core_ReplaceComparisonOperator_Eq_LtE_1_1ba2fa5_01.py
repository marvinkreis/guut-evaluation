from string_utils.validation import is_isbn_13

def test__is_isbn_13():
    """This test ensures that the mutant fails on improperly formatted ISBNs."""
    # Testing a malformed ISBN with a non-integer character
    result_1 = is_isbn_13('978-0-306-40x15-7')  # Should be invalid
    assert not result_1, "The mutant should fail on this malformed ISBN but it passed."

    # Testing another malformed ISBN starting with letters
    result_2 = is_isbn_13('abc-0-306-40615-7')  # Should be invalid
    assert not result_2, "The mutant should fail on this malformed ISBN but it passed."