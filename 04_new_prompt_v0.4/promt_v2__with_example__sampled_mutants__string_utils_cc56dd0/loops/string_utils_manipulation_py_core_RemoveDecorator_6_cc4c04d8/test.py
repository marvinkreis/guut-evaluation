from string_utils.manipulation import __StringCompressor, compress

def test_decompress_mutant_killing():
    """
    Test the decompress method of __StringCompressor. The baseline will correctly perform the decompression
    and return 'Hello, World!', while the mutant, due to missing the @classmethod decorator, will raise a TypeError.
    """
    compressed_string = compress("Hello, World!", encoding='utf-8', compression_level=9)
    
    # Attempt to decompress using the class method
    output = __StringCompressor.decompress(compressed_string)
    assert output == "Hello, World!", f"Expected 'Hello, World!', got {output}"