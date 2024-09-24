from string_utils.generation import roman_range

def test__roman_range_invalid():
    """The mutant version should fail to raise OverflowError for invalid start/stop/step configurations."""
    try:
        roman_range(5, start=4, step=-1)
        assert False, "Expected OverflowError not raised for invalid configuration"
    except OverflowError:
        pass  # Expected outcome