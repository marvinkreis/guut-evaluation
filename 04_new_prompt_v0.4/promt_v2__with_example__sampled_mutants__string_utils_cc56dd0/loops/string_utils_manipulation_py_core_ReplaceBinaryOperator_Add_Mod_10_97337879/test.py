from string_utils.manipulation import prettify

def test_prettify_special_character_dominance_mutant_killing():
    """
    Test the prettify function with a string dominated by special characters and irregular spacing.
    The mutant will raise a TypeError due to the faulty implementation in the __ensure_spaces_around method,
    while the baseline will return a correctly formatted string.
    """
    input_string = '    ***   !!!   ###   '  # Input string with special characters

    # Execute prettify and validate the output on baseline
    output = prettify(input_string)
    assert output == '*  *  * !!! ###', f"Expected '*  *  * !!! ###', got '{output}'"