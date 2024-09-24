from string_utils.manipulation import compress, decompress

def test_compress():
    # Original test string
    original_string = "This is a test string for compression. It should be long enough to benefit from compression."

    # 1. Test valid compression level of 9
    compressed_output = compress(original_string, compression_level=9)
    decompressed_output = decompress(compressed_output)
    assert decompressed_output == original_string, "Decompressed output should match the original string with compression level 9."

    # 2. Test valid compression level of 0
    compressed_output_zero = compress(original_string, compression_level=0)
    decompressed_output_zero = decompress(compressed_output_zero)
    assert decompressed_output_zero == original_string, "Decompressed output should match the original string with compression level 0."

    # 3. Test with invalid compression level of 10
    # Should raise ValueError in the correct implementation
    try:
        compress(original_string, compression_level=10)
        # If no error is raised, we are running the mutant code.
        raise AssertionError("Expected ValueError due to invalid compression level of 10, but no error was raised.")
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9', "Unexpected error message for compression level 10."

    # 4. Test with invalid compression level of -1
    try:
        compress(original_string, compression_level=-1)
        # If no error is raised, we are running the mutant code.
        raise AssertionError("Expected ValueError due to invalid compression level of -1, but no error was raised.")
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9', "Unexpected error message for compression level -1."

# Ensure to run the test
if __name__ == "__main__":
    test_compress()