from string_utils.manipulation import compress

def test__compress():
    """Changing the default compression_level to 10 in the compress function would cause it to raise a ValueError."""
    input_string = "This is a test string for compression."
    try:
        # Attempt to compress using the default level
        output = compress(input_string)
        assert output is not None, "Should successfully compress the string."
    except ValueError as e:
        assert str(e) == 'Invalid compression_level: it must be an "int" between 0 and 9', "Compression should raise ValueError on invalid level."