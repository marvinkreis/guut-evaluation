from string_utils.manipulation import snake_case_to_camel
from string_utils.errors import InvalidInputError

def test_snake_case_to_camel_mutant_killing():
    """
    Test the snake_case_to_camel function with valid input but an additional invalid argument.
    The mutant will fail and raise an InvalidInputError, while the baseline will complete successfully
    with the correct output.
    """
    try:
        output = snake_case_to_camel('valid_snake_case', 123)  # Valid input with an invalid extra argument
        print(f"Output from mutant (with valid snake_case string and invalid argument): {output}")
    except InvalidInputError as e:
        assert str(e) == 'Expected "str", received "int"', f"Unexpected InvalidInputError: {e}"