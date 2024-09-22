from string_utils.manipulation import compress

def test_compress():
    # Test with valid compression level of 0
    try:
        result = compress("test string", compression_level=0)
        assert isinstance(result, str), "Expected valid string output for compression_level=0"
    except Exception as e:
        assert False, f"Did not expect an exception for compression_level=0: {e}"

    # Test with valid compression level of 9
    try:
        result = compress("test string", compression_level=9)
        assert isinstance(result, str), "Expected valid string output for compression_level=9"
    except Exception as e:
        assert False, f"Did not expect an exception for compression_level=9: {e}"

    # Test with invalid compression level of -1
    try:
        compress("test string", compression_level=-1)
        assert False, "Expected ValueError for compression_level=-1"
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9'

    # Test with invalid compression level of 10
    try:
        compress("test string", compression_level=10)
        assert False, "Expected ValueError for compression_level=10"
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9'