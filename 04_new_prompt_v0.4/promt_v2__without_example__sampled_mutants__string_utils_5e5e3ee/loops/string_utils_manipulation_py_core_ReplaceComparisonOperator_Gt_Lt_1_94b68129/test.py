from string_utils.manipulation import compress

def test__compress_valid_compression_level():
    """
    Test whether the compress function works for a valid compression_level of 8.
    The baseline should return a non-empty string, while the mutant should raise a ValueError,
    indicating that the input is invalid.
    """
    try:
        output = compress("test string", compression_level=8)
        assert output != "", "Expected a non-empty output from the baseline."
    except ValueError as e:
        output = str(e)
        assert False, f"Unexpected ValueError raised in baseline: {output}"
    except Exception as e:
        output = str(e)
        assert False, f"Unexpected error raised in baseline: {output}"

    print("Test completed successfully.")