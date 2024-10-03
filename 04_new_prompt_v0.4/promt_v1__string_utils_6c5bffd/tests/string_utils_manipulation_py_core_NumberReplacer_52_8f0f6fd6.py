from string_utils.manipulation import compress

def test__compress_empty_string():
    """
    This test verifies that calling compress with an empty string raises a ValueError in the baseline,
    but does not raise an error in the mutant, demonstrating the mutant's defect.
    """
    try:
        compress('')
    except ValueError as ve:
        assert str(ve) == 'Input string cannot be empty'  # This should pass in the baseline
    except Exception as e:
        assert False, f"Unexpected exception raised on baseline: {e}"  # Should not reach here
    else:
        assert False, "No error raised on baseline, expected ValueError."  # Should not reach here