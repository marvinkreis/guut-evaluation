from string_utils.validation import is_snake_case

def test__is_snake_case():
    """The mutant incorrectly accepts an empty string as a valid snake case string."""
    output = is_snake_case('')
    assert output is False, "An empty string should not be considered a valid snake case."