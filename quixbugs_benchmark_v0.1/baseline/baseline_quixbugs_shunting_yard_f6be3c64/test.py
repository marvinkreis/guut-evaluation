from shunting_yard import shunting_yard

def test__shunting_yard():
    # Test case to detect the mutant
    input_tokens = [3, '+', 4, '*', 2]
    expected_output = [3, 4, 2, '*', '+']  # RPN of 3 + (4 * 2)
    
    actual_output = shunting_yard(input_tokens)
    
    assert actual_output == expected_output, f"Expected {expected_output} but got {actual_output}"