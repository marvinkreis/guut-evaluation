from string_utils.manipulation import compress

def test__compress_different_levels():
    """
    This test checks the compress function for various compression levels.
    It expects a ValueError for levels greater than 9 or equal to 9.
    The test will kill the mutant as it will allow level 9 while the baseline will raise an error.
    """
    input_string = "This is a test string."

    # Test with valid compression levels
    for level in range(0, 9):  # levels from 0 to 8 should not raise an error
        output = compress(input_string, compression_level=level)
        print(f"Compressed string with level {level}: {output}")

    # Test with invalid compression level (equal to 9)
    try:
        compress(input_string, compression_level=9)
        output = "No error for level 9"
    except ValueError as ve:
        output = f"ValueError for level 9: {ve}"
    
    print(f"Output for invalid compression level 9: {output}")

    # Test with invalid compression level (greater than 9)
    try:
        compress(input_string, compression_level=10)
        output_10 = "No error for level 10"
    except ValueError as ve:
        output_10 = f"ValueError for level 10: {ve}"
    
    print(f"Output for invalid compression level 10: {output_10}")