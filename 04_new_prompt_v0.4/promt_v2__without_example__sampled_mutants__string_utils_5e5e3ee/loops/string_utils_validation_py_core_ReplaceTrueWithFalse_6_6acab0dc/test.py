from string_utils.validation import is_isbn

def test__is_isbn_with_hyphens():
    """
    Test that the ISBN with hyphens is handled correctly.
    The input '978-0312498580' should return True in the baseline,
    but False in the mutant due to the normalization parameter change.
    """
    output = is_isbn('978-0312498580')
    assert output == True, f"Expected True, got {output}"