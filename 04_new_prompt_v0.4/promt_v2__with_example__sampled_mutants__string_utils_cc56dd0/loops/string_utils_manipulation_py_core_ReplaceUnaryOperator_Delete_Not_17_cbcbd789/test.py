from string_utils.manipulation import strip_margin
from string_utils.errors import InvalidInputError  # Ensure we can validate appropriately

def test_strip_margin_functionality():
    """
    Test the strip_margin function to confirm its behavior with valid and invalid inputs.
    The baseline should raise InvalidInputError for invalid inputs, while the mutant should allow further processing,
    leading to unexpected behavior.
    """

    # Valid input test
    valid_input = """
                line 1
                line 2
                line 3
                """
    output_valid = strip_margin(valid_input)
    assert output_valid.strip() == "line 1\nline 2\nline 3", f"Expected formatted output, but got: {output_valid}"

    # Invalid integer input test
    try:
        strip_margin(12345)
        assert False, "Expected an exception for integer input, but none was raised."
    except InvalidInputError:
        print("Caught expected InvalidInputError for integer input.")

    # Invalid list input test
    try:
        strip_margin(["line1", "line2"])
        assert False, "Expected an exception for list input, but none was raised."
    except InvalidInputError:
        print("Caught expected InvalidInputError for list input.")
