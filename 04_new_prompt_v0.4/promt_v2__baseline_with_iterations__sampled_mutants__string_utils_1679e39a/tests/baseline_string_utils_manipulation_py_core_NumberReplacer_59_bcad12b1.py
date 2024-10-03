from string_utils.manipulation import compress

def test__compress():
    """
    Test that verifies if the `compress` function does not raise a ValueError for a valid compression level of 9.
    The baseline allows valid ranges of compression levels between 0 and 9, while the mutant restricts it to between 0 and 8.
    Therefore, passing a value of 9 should succeed on the baseline but raise an error on the mutant.
    """
    output = compress("sample input", compression_level=9)
    assert isinstance(output, str), "Output should be a string."