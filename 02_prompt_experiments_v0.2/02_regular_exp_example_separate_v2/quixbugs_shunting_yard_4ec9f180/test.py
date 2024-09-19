from shunting_yard import shunting_yard

def test__shunting_yard():
    """Removing the operator addition line prevents shunting_yard from creating valid RPN."""
    input_tokens = [10, '-', 5]
    output = shunting_yard(input_tokens)
    # Expected output for this case
    expected_output = [10, 5, '-']
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

    input_tokens_complex = [4, '+', 9, '*', 9, '-', 10, '+', 13]
    output_complex = shunting_yard(input_tokens_complex)
    # Expected output for the complex case
    expected_output_complex = [4, 9, 9, '*', '+', 10, '-', 13, '+']
    assert output_complex == expected_output_complex, f"Expected {expected_output_complex}, but got {output_complex}"