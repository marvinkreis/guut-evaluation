from string_utils.validation import is_isbn_13

def test__isbn_13_invalid_length():
    """
    Test to confirm that the is_isbn_13 function correctly rejects strings that are not valid ISBN-13.
    The input "97803124985800" is longer than 13 characters and thus should return False for both
    Baseline and Mutant versions. Specifically, we expect the Baseline to pass (return False),
    whereas the Mutant should fail (return True).
    """
    output = is_isbn_13("97803124985800")
    assert output == False