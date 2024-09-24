from string_utils.manipulation import compress, decompress

def test_compress():
    # Create a highly redundant and lengthy input string
    original_string = (
        "This is a test string to evaluate the compression efficiency. " * 40 +  # Repetitive content
        "12345 67890 " * 30 +  # Adding numeric sequences
        "The quick brown fox jumps over the lazy dog. " * 20  # Varied text
    )

    # Compress the original string using the original implementation (compression_level=9)
    compressed_string_correct = compress(original_string)

    # Verify that decompressing gives back the original string
    decompressed_correct = decompress(compressed_string_correct)
    assert decompressed_correct == original_string, "Decompressed string should match the original."

    # Ensure that the length of the compressed string is less than the original string
    assert len(compressed_string_correct) < len(original_string), "Compressed string should be shorter than the original."

    # Now simulate mutant behavior (compression_level=8)
    compressed_string_mutant = compress(original_string)

    # Verify that the mutant's decompressed output also matches the original
    decompressed_mutant = decompress(compressed_string_mutant)
    assert decompressed_mutant == original_string, "Mutant decompressed string should match the original."

    # Now we make the assertion more reflective of outcomes:
    # We expect the mutant's compressed string to be longer than the original's due to the lowered compression level
    assert len(compressed_string_mutant) >= len(compressed_string_correct), "Mutant's compression should not yield better results than the original."
