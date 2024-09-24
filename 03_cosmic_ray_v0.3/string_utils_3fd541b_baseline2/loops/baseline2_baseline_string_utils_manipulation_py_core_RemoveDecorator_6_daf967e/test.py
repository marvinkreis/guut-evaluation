from string_utils.manipulation import compress, decompress

def test__decompress_removal():
    original_string = "This is a test string."
    # First, compress the string
    compressed = compress(original_string)
    # Now, attempt to decompress the string
    decompressed = decompress(compressed)
    # Assert that the original string and the decompressed string are the same
    assert original_string == decompressed