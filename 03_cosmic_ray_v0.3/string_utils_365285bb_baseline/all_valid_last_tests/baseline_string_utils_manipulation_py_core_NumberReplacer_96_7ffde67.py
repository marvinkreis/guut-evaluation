from string_utils.manipulation import compress, decompress

def test__compress_mutant_detection():
    original_string = 'This is a test string for compression.'
    
    # Test with valid compression levels (0 through 9)
    
    # Compression Level 0
    compressed_string_0 = compress(original_string, compression_level=0)
    decompressed_string_0 = decompress(compressed_string_0)
    assert decompressed_string_0 == original_string, "Decompressed string must match the original for level 0."

    # Compression Level 1
    compressed_string_1 = compress(original_string, compression_level=1)
    decompressed_string_1 = decompress(compressed_string_1)
    assert decompressed_string_1 == original_string, "Decompressed string must match the original for level 1."
    
    # Compression Level 9
    compressed_string_9 = compress(original_string, compression_level=9)
    decompressed_string_9 = decompress(compressed_string_9)
    assert decompressed_string_9 == original_string, "Decompressed string must match the original for level 9."

    # SECTION: Testing invalid compression level,
    # for original implementation that should raise ValueError
    try:
        # This should raise a ValueError for the original implementation
        compress(original_string, compression_level=10)
        assert False, "Expected ValueError for compression level 10 in the original code."
    except ValueError:
        pass  # This is expected behavior in the original implementation

    # Now we need to focus on mutant behavior with a valid level
    # Instead of invoking level 10, we might also want to test with the max valid level
    # which should be recognized by both implementations (level 9)
    # This will test if mutant behaves similarly with valid inputs.

    # Check the mutants expectations
    compressed_mutant_string = compress(original_string, compression_level=9)  # Should work

    # Check the output should be non-empty
    assert compressed_mutant_string != "", "Mutant should produce a non-empty compressed output for level 9."

    # Attempt to decompress the mutant's output
    decompressed_mutant_string = decompress(compressed_mutant_string)

    # Ensure that decompressed mutant output meets expectations.
    assert decompressed_mutant_string == original_string, "Mutant should yield the original string for valid compression."
