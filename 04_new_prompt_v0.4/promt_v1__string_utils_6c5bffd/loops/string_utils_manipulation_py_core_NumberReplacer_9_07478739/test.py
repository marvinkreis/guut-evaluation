from string_utils.manipulation import roman_encode

def test__roman_encode():
    """
    Test the roman_encode function to ensure that it correctly converts integers to Roman numerals.
    Specifically, we are testing numbers that include hundreds to trigger the mutation. 
    The mutant has an incorrect mapping for hundreds that results in a KeyError.
    
    The expected outputs are:
    - 100 should return 'C'
    - 200 should return 'CC'
    - 300 should return 'CCC'
    - 400 should return 'CD'
    - 500 should return 'D'
    """
    inputs = [100, 200, 300, 400, 500]
    expected_outputs = ['C', 'CC', 'CCC', 'CD', 'D']

    for i, input_number in enumerate(inputs):
        output = roman_encode(input_number)
        assert output == expected_outputs[i], f"Expected {expected_outputs[i]} but got {output} for input {input_number}"