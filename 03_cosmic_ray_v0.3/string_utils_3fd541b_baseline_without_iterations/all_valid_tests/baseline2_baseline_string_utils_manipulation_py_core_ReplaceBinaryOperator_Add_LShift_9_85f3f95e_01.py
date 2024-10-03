from string_utils.manipulation import prettify

def test__prettify_invalid_spaces_around():
    # Test case for `prettify` that will hit the mutant
    input_string = "hello world"
    expected_output = "Hello world"  # Prettified output expected from the correct implementation

    # The mutant will raise a SyntaxError when trying to prettify this string
    try:
        prettified_output = prettify(input_string)
        assert prettified_output == expected_output, f"Expected '{expected_output}', but got '{prettified_output}'"
    except SyntaxError:
        print("Mutant detected: SyntaxError occurred as expected.")