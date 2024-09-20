from string_utils.manipulation import compress

def test__compress():
    """Changing the default compression_level to 10 in the compress function would cause it to raise a ValueError."""
    input_string = "This is a test string for compression."

    # Verify that compress successfully runs with correct code
    output = compress(input_string)
    assert output is not None, "Should successfully compress the string."
    
    # Check the behavior when compressing with the mutant settings (the default value is now invalid)
    try:
        # Simulate what would happen if the compression level was invalid
        mutant_output = compress(input_string, compression_level=10)
        assert False, "mutant compress should have raised a ValueError due to invalid compression level."
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9', "Expected ValueError not raised properly."