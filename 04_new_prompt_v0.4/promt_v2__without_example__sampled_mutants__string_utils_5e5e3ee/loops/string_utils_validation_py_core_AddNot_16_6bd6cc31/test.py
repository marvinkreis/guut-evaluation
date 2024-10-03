from string_utils.validation import is_snake_case

def test__is_snake_case_mutant_killer():
    """
    Test that checks if the input "valid_snake_case" is identified correctly as a valid snake case string.
    The baseline should return True, but the mutant should return False, thus killing the mutant.
    """
    output = is_snake_case("valid_snake_case")
    assert output == True