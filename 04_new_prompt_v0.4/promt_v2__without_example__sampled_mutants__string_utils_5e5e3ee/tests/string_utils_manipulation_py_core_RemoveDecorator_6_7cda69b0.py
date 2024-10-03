from string_utils.manipulation import compress, decompress

def test__compress_decompress():
    """
    Test the compress and decompress functions to ensure they behave correctly. 
    The input 'Hello World!' should be compressed and then successfully decompressed to yield the original string.
    The mutant is expected to fail due to an AttributeError when trying to call the decompression method.
    """
    original_string = 'Hello World!'
    compressed_string = compress(original_string)
    decompressed_string = decompress(compressed_string)
    
    assert decompressed_string == original_string, f"Expected: {original_string}, but got: {decompressed_string}"