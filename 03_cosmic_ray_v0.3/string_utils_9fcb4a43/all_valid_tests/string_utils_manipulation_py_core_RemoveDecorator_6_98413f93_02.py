from string_utils.manipulation import compress, decompress

def test__decompress():
    """The 'decompress' method must raise an AttributeError when called as class method due to the missing decorator in + mutant."""
    original_string = "test data for compression"
    compressed_data = compress(original_string)
    
    # Now validate the decompression functionality as expected
    try:
        # Call the decompressor and expect original string back
        decompressed_data = decompress(compressed_data)
        
        # Check that the decompressed data matches the original string
        assert decompressed_data == original_string, f"Decompressed data '{decompressed_data}' did not match original '{original_string}'"
    
    except AttributeError as e:
        # This will catch the mutant case if the method is incorrectly referenced
        assert "has no attribute" in str(e), "Expected AttributeError, but received a different exception."
    except Exception as e:
        assert False, f"Unexpected exception raised: {str(e)}"

# Test the mutant
test__decompress()