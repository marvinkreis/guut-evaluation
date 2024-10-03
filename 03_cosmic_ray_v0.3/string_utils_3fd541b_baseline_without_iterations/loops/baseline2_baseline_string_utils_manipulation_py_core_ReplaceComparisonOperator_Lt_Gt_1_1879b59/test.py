from string_utils.manipulation import compress

def test__compress_invalid_level():
    # Test case where the compression level is invalid
    try:
        # Passing -1 as compression level should raise ValueError
        compress("test string", compression_level=-1)
        assert False, "Expected ValueError for compression level -1"
    except ValueError:
        pass  # Correct behavior, ValueError is raised

    try:
        # Passing 10 as compression level should raise ValueError
        compress("test string", compression_level=10)
        assert False, "Expected ValueError for compression level 10"
    except ValueError:
        pass  # Correct behavior, ValueError is raised

    try:
        # Passing 0 as compression level should not raise an error
        result = compress("test string", compression_level=0)
        assert isinstance(result, str)  # Check if the returned result is indeed a string
    except ValueError:
        assert False, "Did not expect ValueError for compression level 0"

    # Test case with valid compression level
    try:
        result = compress("test string", compression_level=5)
        assert isinstance(result, str)  # Check if the returned result is indeed a string
    except ValueError:
        assert False, "Did not expect ValueError for valid compression level 5"