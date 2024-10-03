from string_utils.validation import is_isbn_10, is_isbn_13

def test_is_isbn_killing_mutant():
    """
    Test the ISBN validation methods with inputs that contain hyphens.
    The baseline should accept these inputs as valid ISBNs, while the mutant will fail.
    """
    # Test input for is_isbn_10
    assert is_isbn_10('150-6715214', normalize=True) == True, "Expected True for ISBN-10"

    # Test input for is_isbn_13
    assert is_isbn_13('978-0312498580', normalize=True) == True, "Expected True for ISBN-13"