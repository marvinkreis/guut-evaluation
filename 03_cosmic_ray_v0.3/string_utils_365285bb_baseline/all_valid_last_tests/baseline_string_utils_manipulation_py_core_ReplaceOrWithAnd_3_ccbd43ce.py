from string_utils.manipulation import compress, decompress

def test_compress():
    # Define a string for testing
    original_string = "This is a test string for compression."

    # Try compressing using an invalid compression level (e.g., 10)
    try:
        compress(original_string, compression_level=10)
        assert False, "Expected ValueError for invalid compression level (10) but got no exception."
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9'

    # Compress with a valid compression level
    compressed_string = compress(original_string, compression_level=5)

    # Decompress the string
    decompressed_string = decompress(compressed_string)

    # Check that the decompressed string is equal to the original string
    assert decompressed_string == original_string

    # Check that the mutant's logic incorrectly allows invalid compression
    try:
        compress(original_string, compression_level=-1)
        assert False, "Mutant should have raised a ValueError for invalid compression level (-1) but did not."
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9'