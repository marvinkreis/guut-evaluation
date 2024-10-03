from string_utils.manipulation import compress

def test__compress():
    """
    Test whether the compression level can be set to 0. The original code allows a compression level of 0,
    while the mutant changes this to disallow it, resulting in an error. This test case ensures that when using
    a compression level of 0, the original code completes successfully, but the mutant raises a ValueError.
    """
    output = compress("test string", compression_level=0)
    assert output is not None  # This should not raise an error and should return a result