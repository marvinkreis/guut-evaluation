from string_utils.manipulation import decompress, compress

def test__decompress_kills_mutant():
    """
    This test ensures that the `decompress` method behaves as expected with the Baseline
    and fails with the Mutant due to the absence of the @classmethod decorator.
    A valid base64 compressed input is used to ensure that the input is appropriate.
    The expected output is 'Hello, World!', confirming functionality in the Baseline.
    """
    
    original_string = "Hello, World!"  # Original string for compression
    compressed_string = compress(original_string)  # Creating a known valid compressed string
    
    # First, test on the Baseline
    result = decompress(compressed_string)
    assert result == original_string, f"Expected '{original_string}' but got '{result}'"

    # Now, test on the Mutant - expecting it to raise an error
    try:
        result_mutant = decompress(compressed_string)
        print("Mutant did not raise an error.")
    except Exception as e:
        print(f"Mutant output = {e}")

test__decompress_kills_mutant()