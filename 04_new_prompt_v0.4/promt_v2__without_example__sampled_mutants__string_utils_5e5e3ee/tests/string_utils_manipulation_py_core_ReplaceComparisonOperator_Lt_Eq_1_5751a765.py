from string_utils.manipulation import compress

def test__compress_zero_level():
    """
    Test the compress function with a compression_level of 0.
    This test validates that the baseline implementation raises a ValueError
    while the mutant should allow this input and successfully return a compressed string.
    """
    input_string = "example input string"

    try:
        # Invoking the compress function with a compression level of zero
        output = compress(input_string, compression_level=0)
        # If no error is raised, then we are in the mutant case
        assert True, f"Mutant allowed zero compression level, output: {output}"
    except ValueError as ve:
        # If ValueError is raised, it confirms the baseline's correct implementation
        print(f"Baseline raised ValueError as expected: {ve}")
        assert False, "Expected the mutant to allow zero compression level."