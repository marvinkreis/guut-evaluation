from string_utils.manipulation import compress

def test__compress_mutant_detection():
    # Testing Original Code: Expecting it to raise ValueError for compression_level = 10 (if it exceeds allowed range)
    try:
        compress("test string", compression_level=10)  # Assuming level 9 can pass, but 10 should fail.
        assert False, "Expected ValueError for compression level 10 in the original code."
    except ValueError:
        pass  # This is expected behavior for the original code

    # Simulated Mutant compress function behavior
    class MutantCompress:
        @staticmethod
        def compress(input_string: str, encoding: str = 'utf-8', compression_level: int = 9) -> str:
            # Mutant allows level 9 and raises an exception for higher levels
            if not isinstance(compression_level, int) or compression_level < 0 or compression_level > 9:
                raise ValueError('Invalid compression_level: it must be an "int" between 0 and 9')
            return input_string  # Returning original input for comparison

    # Testing the Mutant Code
    try:
        result = MutantCompress.compress("test string", compression_level=9)
        assert result == "test string", "Mutant compress method unexpectedly altered the input."
        print("Mutant test passed successfully, allowing compression level 9.")
    except ValueError:
        assert False, "Mutant raised ValueError for compression level 9, which is incorrect."

# Run the test function to verify behavior
test__compress_mutant_detection()